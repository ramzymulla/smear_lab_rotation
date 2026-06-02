import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import keypoint_moseq as kpms
import tqdm
import h5py
import matplotlib.pyplot as plt
import sys
import numpy as np
import itertools
from pathlib import Path
import jax_moseq


def plot_param_scan(params, project_dir, prefix, paramName, figsize=(8, 2.5)):
    """Plot the results of a param scan.

    This function assumes that model results for each kappa value are stored
    in `{project_dir}/{prefix}-{param}/checkpoint.h5`. Two plots are generated:
    (1) a line plot showing the median syllable duration over the course of
    fitting for each kappa value; (2) and a plot showing the final median
    syllable duration as a function of kappa.

    Parameters
    ----------
    kappas : array-like of float
        Kapppa values used in the scan.

    project_dir : str
        Path to the project directory.

    prefix : str
        Prefix for the kappa scan model names.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.

    final_median_durations : array of float
        Median syllable durations for each kappa value, derived using the
        final iteration of each model.
    """
    median_dur_histories = []
    final_median_durs = []

    for param in tqdm.tqdm(params, desc="Loading checkpoints"):
        model_dir = f"{project_dir}/{prefix}-{paramName}{param}"
        with h5py.File(f"{model_dir}/checkpoint.h5", "r") as h5:
            mask = h5["data/mask"][()]
            iterations = np.sort([int(i) for i in h5["model_snapshots"]])

            history = {}
            for itr in iterations:
                z = h5[f"model_snapshots/{itr}/states/z"][()]
                durs = kpms.get_durations(z, mask)
                history[itr] = np.median(durs)

            final_median_durs.append(history[iterations[-1]])
            median_dur_histories.append(history)

    fig, axs = plt.subplots(1, 2)
    for i, (param, history) in enumerate(zip(params, median_dur_histories)):
        color = plt.cm.viridis(i / (len(params) - 1))
        label = "{:.1e}".format(param)
        axs[0].plot(*zip(*history.items()), color=color, label=label)
    axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("median duration")

    axs[1].scatter(params, final_median_durs)
    axs[1].set_xlabel(paramName)
    axs[1].set_ylabel("final median duration")
    axs[1].set_xscale("log")

    fig.set_size_inches(figsize)
    plt.tight_layout()
    return fig, np.array(final_median_durs)

def main():

    framerate = sys.argv[1]
    sleap_dir = sys.argv[2]
    includeStr = sys.argv[3].split('/')[-1]


    rslds=True if 'rslds' in sys.argv else False
    if rslds:
        from jax_moseq.models.keypoint_slds.decision_list import greedy_decision_list_permutation, permute_model_states
        project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/rslds_keypoint_moseq_{framerate}b/"))
        jitterAR=1e-1
        jitterSLDS=1e-1
        kappaAR=1e12
        kappaSLDS=1e6
        lWs = np.logspace(-4,2,7)
        lbs = np.logspace(-4,2,7)

    else:
        project_dir = str(Path(f"/home/rza/Research/smearlab/scripts/kpms_outputs/keypoint_moseq_{framerate}/"))
        jitterAR=1e-1
        jitterSLDS=1e-1
        kappaAR=1e12
        kappaSLDS=5*np.logspace(2,10)
        lW = 0.1
        lb = 0.05

    base_path_str = "/home/rza/Research/smearlab/clickbait-loco/thermister/"
    models_dir = "/home/rza/Research/smearlab/scripts/sleap/"

    sleap_files = [str(i) for i in list(Path(sleap_dir).glob(includeStr.replace('.avi','.slp')))]
    video_dir = str(Path(base_path_str))  # any directory containing videos (e.g. .mp4, .avi) and/or sleap files with predictions for those videos
    if not os.path.exists(project_dir):
        kpms.setup_project(project_dir, sleap_file=sleap_files[0],video_dir=video_dir) 
    else:
        kpms.update_config(project_dir,
                        sleap_file=sleap_files[0],
                        video_dir=video_dir)
        
    config = lambda: kpms.load_config(project_dir)

    # load data 
    keypoint_data_path = sleap_files  # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")


    # update config
    kpms.update_config(project_dir,
                    use_bodyparts=['nose1','neck1','centroid1','forepawL1','forepawR1','hindpawL1','hindpawR1','tailstart1'],
                    anterior_bodyparts=["nose1","neck1"],posterior_bodyparts=["tailstart1"],fps=int(framerate))


    ### PREPROCESSING ###
    coordinates, confidences = kpms.outlier_removal(
        coordinates,
        confidences,
        project_dir,
        overwrite=False,
        **config()
    )

    data, metadata = kpms.format_data(coordinates, confidences, **config())
    pca = kpms.fit_pca(**data, **config())

    kpms.save_pca(pca, project_dir)

    cs = np.cumsum(pca.explained_variance_ratio_)

    latent_dim = min((cs>0.9).nonzero()[0].min()+1,10)

    kpms.update_config(project_dir, latent_dim=int(latent_dim))
    print(f"latent dim = {latent_dim}")

    # kpms.update_config(
    #     project_dir,
    #     sigmasq_loc=kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config()["fps"])
    # )
    data = jax_moseq.utils.debugging.convert_data_precision(data)

    ### MODEL FITTING ###

    # initialize the model
    model = kpms.init_model(data, pca=pca, **config())

    # # optionally modify kappa
    model = kpms.update_hypparams(model, kappa=kappaAR)

    num_ar_iters = 50
    num_slds_iters = 200
    ar_model_name = "param_sweep_AR"
    ar_model, ar_model_name = kpms.fit_model(
        model, data, metadata, project_dir, 
        model_name=ar_model_name, jitter=jitterAR,ar_only=True, num_iters=num_ar_iters
    )

    # load model checkpoint
    ar_model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir, ar_model_name, iteration=num_ar_iters
    )

    if rslds: 
        # Extract optimal ordering and permute model based on settled SLDS latents
        num_states = ar_model["hypparams"]["trans_hypparams"]["num_states"]

        optimal_permutation, R_init, r_init = greedy_decision_list_permutation(
            ar_model["states"]["x"], 
            ar_model["states"]["z"], 
            num_states
        )

    for indw,lW in enumerate(lWs):
        for indb,lb in enumerate(lbs):
            model, data, metadata, current_iter = kpms.load_checkpoint(
                                        project_dir, ar_model_name, iteration=num_ar_iters
                                    )
            model = permute_model_states(model, optimal_permutation, R_init, r_init)

            # # modify kappa to maintain the desired syllable time-scale
            model = kpms.update_hypparams(model, kappa=kappaSLDS)

            model['hypparams']['trans_hypparams']['lambda_W'] = lW
            model['hypparams']['trans_hypparams']['lambda_b'] = lb


            # run SLDS/rSLDS fitting for an additional 500 iters
            model, model_name = kpms.fit_model(
                model,
                data,
                metadata,
                project_dir,
                model_name=f"param_sweep_lW{lW}-lb{lb}".replace("+",""),
                jitter=jitterSLDS,
                ar_only=False,
                start_iter=num_ar_iters,
                num_iters=num_ar_iters+num_slds_iters,
                save_every_n_iters=25
            )

            ### VISUALIZE & SAVE MODEL FIT RESULTS ###

            # modify a saved checkpoint so syllables are ordered by frequency
            kpms.reindex_syllables_in_checkpoint(project_dir, model_name);

            # load the most recent model checkpoint
            model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

            # extract results
            results = kpms.extract_results(model, metadata, project_dir, model_name)

            kpms.save_results_as_csv(results, project_dir, model_name)

            kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())

            try:
                kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())
            except:
                kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, min_frequency=0.01, **config())

            plt.close('all')
            # kpms.generate_grid_movies(results, project_dir, model_name, 
            #                         overlay_keypoints=True,coordinates=coordinates, **config());

        fig,_ = plot_param_scan(lbs,project_dir,f"param_sweep_lW{lW}","lb")
        fig.savefig(os.path.join(project_dir,f"param_scan_lW{lW}.png"), format='png',dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    main()
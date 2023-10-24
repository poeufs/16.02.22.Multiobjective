if __name__ = = __main__():
    # Change directory if needed
    if os.getcwd() != project_dir:
        print(f'Current directory is {os.getcwd()}')
        os.chdir(project_dir)
        print(f'Changed current directory to {project_dir}')

    # Initialise the archives list
    all_archives = []

    # Load the archives
    for i in range(seeds):
        archives = ArchiveLogger.load_archives(f"{archives_dir_path}/{i}.tar.gz")
        all_archives.append(archives)

    # Define the problem
    problem = to_problem(model, searchover="levers")
    print('problem type is', type(problem))

    # Define the reference list
    reference_set = epsilon_nondominated(results_list, [0.8] * len(model.outcomes), problem) #[0.05]
    print('reference_set', reference_set)
    print('reference_set type is', type(reference_set))


    hv = HypervolumeMetric(reference_set, problem)

    metrics_by_seed = []
    for archives in all_archives:
        metrics = []
        for nfe, archive in archives.items():
            scores = {
                "hypervolume": hv.calculate(archive),
                "nfe": int(nfe),
            }
            metrics.append(scores)
        metrics = pd.DataFrame.from_dict(metrics)

        # sort metrics by number of function evaluations
        metrics.sort_values(by="nfe", inplace=True)
        metrics_by_seed.append(metrics)

    # Visualize convergence metrics
    sns.set_style("white")
    fig, axes = plt.subplots(nrows=2, figsize=(8, 12), sharex=True)

    ax1, ax2 = axes

    for metrics, convergence in zip(metrics_by_seed, convergences):
        ax1.plot(metrics.nfe, metrics.hypervolume)
        ax1.set_ylabel("hypervolume")

        ax2.plot(convergence.nfe, convergence.epsilon_progress)
        ax2.set_ylabel("$\epsilon$ progress")

        '''
        ax3.plot(metrics.nfe, metrics.generational_distance)
        ax3.set_ylabel("generational distance")

        ax4.plot(metrics.nfe, metrics.epsilon_indicator)
        ax4.set_ylabel("epsilon indicator")

        ax5.plot(metrics.nfe, metrics.inverted_gd)
        ax5.set_ylabel("inverted generational\ndistance")

        ax6.plot(metrics.nfe, metrics.spacing)
        ax6.set_ylabel("spacing")
        '''

    # ax6.set_xlabel("nfe")

sns.despine(fig)

plt.show()

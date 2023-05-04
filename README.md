# PA-Maliboo (Parallel Asynchronous Maliboo)
This library implements parallel asynchronous Bayesian Optimization (BO) for tasks running via job schedulers, e.g. on High-Perfomance Computing (HPC) systems.
It is the parallel extension of the [MALIBOO](https://github.com/brunoguindani/MALIBOO) algorithm (MAchine Learning In Bayesian OptimizatiOn).
These algorithms integrate BO with Machine Learning techniques in order to better guide the exploration process of BO.
One can also use the base parallel asynchronous BO algorithm, which employs a Constant Liar approach.

The library is extremely modular and extensible: you can easily build classes for your own objective functions, schedulers, and even acquisition functions.
It is suited for both constrained and unconstrained optimization.

The code already includes compatibility with the [HyperQueue](https://github.com/It4innovations/hyperqueue) meta-scheduler.
HyperQueue is provided as a single, statically linked binary without any dependencies, and is therefore an excellent candidate for a lightweight, self-contained scheduler.
(Versions 0.14.0 and 0.15.0 were used for testing.)

This library supports fault tolerance, as information is constantly saved to human-readable-format files as checkpoint.
Therefore, in case of system crash, the same script can be launched again and the execution of the parallel algorithm will resume as normal.


## Installation
1) Download or clone this repository
2) Install dependencies in `requirements.txt`, or set up a virtual environment
3) Download the [HyperQueue](https://github.com/It4innovations/hyperqueue/releases/latest) meta-scheduler and place it in the `lib` folder
4) Now you can move on to the...


## Tutorial
Run `main_example.py` for a basic example of parallel asynchronous BO on a dummy objective function.
Run `batch_example.py` for an example of batch execution, which creates the database of initial points used by the above script.


## Acknowledgments
This library is currently maintained by the LIGATE project, which was partially funded by the European Commission under the Horizon 2020 Grant Agreement number 956137, as part of the European High-Performance Computing (EuroHPC) Joint Undertaking program.

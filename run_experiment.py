import numpy as np
import os
from copy import deepcopy
import multiprocessing as mp
import time

import data.utils
from sequence_learners.sequence_learner import GeneralParameters
from sequence_learners.informed_guess import InformedGuess
from sequence_learners.local_neighbors import LocalNeighbors, LocalNeighborsParameters
from sequence_learners.rnn import RNN, RNNParameters
from sequence_learners.temporal_mapping import TemporalMapping, TemporalMappingParameters
from sequence_learners.correlation_matrix import CorrelationMatrix, CorrelationMatrixParameters
from sequence_learners.mutual_mappping import MutualMapping, MutualMappingParameters
from sequence_learners.embedding_lstm import EmbeddingLSTM, EmbeddingLSTMParameters
import eval


n_gpus = 4
n_processes = 2

datasets = []
datasets.extend([fname[:-4] for fname in os.listdir("./data/") if fname.startswith("synthetic") and fname.endswith(".pkl")])
datasets.extend(["Financial1", "Financial2"])


def one_experiment(learner, seq_name, parameters, semaphore=None, process_id=None):
    if os.path.exists(parameters["result_path"]):
        return
    train, test = data.utils.load_seq(seq_name)

    if process_id is not None:
        parameters["cuda_id"] = process_id % n_gpus
    else:
        parameters["cuda_id"] = 0

    learner = learner(train, parameters)

    pred = learner.predict(test, parameters)

    for accu_type in ["occurrence", "binary"]:
        accu = eval.calc_accuracy(pred[:-1], test[1:], 128, accu_type, parameters["cuda_id"])
        result_path = parameters["result_path"].split("/")
        result_path[-3] += "_" + accu_type
        result_path = os.path.join(*result_path)
        eval.save_accuracy(result_path, accu)

    if semaphore is not None:
        semaphore.release()
    return


def experiments_for_one_dataset(seq_name, processes=None, semaphore=None):
    global process_id
    result_path = "./results/" + seq_name + "/"
    general_parameters = GeneralParameters(top_k=10, freq_threshold=-1)._asdict()
    general_parameters.update({"cuda_id": 0})
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    '''
    # informed guess
    parameters = deepcopy(general_parameters)
    parameters.update({
        "result_path": result_path + "informed_guess.csv"
    })
    print(parameters["result_path"])
    one_experiment(InformedGuess, seq_name, parameters)

    # local neighbors
    for next_probability in [i/10. for i in range(11)]:
        parameters = deepcopy(general_parameters)
        parameters.update(LocalNeighborsParameters(
            next_probability=next_probability
        )._asdict())
        parameters.update(dict(result_path=result_path+"local_nextprob_"+str(next_probability)+".csv"))
        print(parameters["result_path"])
        one_experiment(LocalNeighbors, seq_name, parameters)

    # correlation matrix
    for kernel_window in [32]:
        parameters = deepcopy(general_parameters)
        parameters.update(CorrelationMatrixParameters(
            kernel_mode="linear",
            kernel_window=kernel_window,
            kernel_diff=10.
        )._asdict())
        parameters.update({"result_path": result_path+"corrmat_kwin_"+str(kernel_window)+".csv"})
        print(parameters["result_path"])
        one_experiment(CorrelationMatrix, seq_name, parameters)
    # temporal mapping
    for kernel_window in [64]:
        for alpha in [5e-1]: # [1e0, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2]:
            for n_dims in [2, 4, 8]:
                semaphore.acquire()
                parameters = deepcopy(general_parameters)
                parameters.update(TemporalMappingParameters(
                    kernel_mode="linear",
                    kernel_window=kernel_window,
                    kernel_diff=10.,
                    alpha=alpha,
                    stop_criterion=alpha * 4e-4 * np.sqrt(n_dims),
                    n_epochs=5000,
                    n_dims=n_dims,
                    norm="std",
                    beta=None,
                    min_repulse_distance=None,
                    n_neighbors=None
                )._asdict())
                parameters.update(dict(result_path=result_path+"tmap_std__kwin_"+str(kernel_window)+
                                                   "_alpha_"+str(alpha)+"_ndims_"+str(n_dims)+".csv"))
                print(parameters["result_path"])
                processes.append(mp.Process(
                    target=one_experiment,
                    args=(TemporalMapping, seq_name, parameters, semaphore, process_id)
                ))
                process_id += 1
                processes[-1].start()
    '''
    # temporal mapping force
    for kernel_window in [64]:
        for alpha in [5e-1]:
            for n_dims in [2]:
                parameters = deepcopy(general_parameters)
                parameters.update(TemporalMappingParameters(
                    kernel_mode="linear",
                    kernel_window=kernel_window,
                    kernel_diff=10.,
                    alpha=alpha,
                    stop_criterion=1e-3,
                    n_epochs=50000,
                    n_dims=n_dims,
                    norm="force",
                    beta=5e-7,
                    gamma=-2,
                    min_repulse_distance=1e-3,
                    n_neighbors=1000,
                )._asdict())
                parameters.update(dict(result_path=result_path+"tmap_force_kwin_"+str(kernel_window)+
                                                   "_alpha_"+str(alpha)+"_ndims_"+str(n_dims)+".csv"))
                print(parameters["result_path"])
                one_experiment(TemporalMapping, seq_name, parameters)
    '''
    # embedding LSTM for address
    for embedding_dim in [256]:
        for sentence_len in [32]:
            semaphore.acquire()
            parameters = deepcopy(general_parameters)
            parameters.update(RNNParameters(
                embedding_dim=embedding_dim,
                hidden_dim=256,
                sentence_len=sentence_len,
                step_len = 7,
                batch_size=256,
                n_epochs=2000,
                max_ele=50000,
                learning_rate=2e-3,
                freq_threshold=None,
            )._asdict())
            parameters.update(dict(result_path=result_path+"lstm_embeddim_"+str(embedding_dim)+
                                   "_sentencelen_"+str(sentence_len)+".csv"))
            print(parameters["result_path"])
            processes.append(mp.Process(
                target=one_experiment,
                args=(RNN, seq_name, parameters, semaphore, process_id)
            ))
            process_id += 1
            processes[-1].start()
    
    # mutual mapping
    semaphore.acquire()
    parameters = deepcopy(general_parameters)
    tmap = TemporalMappingParameters(
        kernel_mode="linear",
        kernel_window=32,
        kernel_diff=10.,
        alpha=0.5,
        stop_criterion=0.5 * 4e-4 * np.sqrt(8),
        n_epochs=5000,
        n_dims=8,
        norm="std",
        beta=None,
        min_repulse_distance=None,
        n_neighbors=None
    )._asdict()
    parameters.update(MutualMappingParameters(
        tmap=tmap,
        hidden_dim=256,
        n_epochs=10000,
        learning_rate=2e-4,
        batch_size=256,
    )._asdict())
    parameters.update(dict(result_path=result_path + "mmap.csv"))
    print(parameters)
    processes.append(mp.Process(
        target=one_experiment,
        args=(MutualMapping, seq_name, parameters, semaphore, process_id)
    ))
    process_id += 1
    processes[-1].start()

    # embedding LSTM for delta
    for embedding_dim in [128]:
        for sentence_len in [64]:
            semaphore.acquire()
            parameters = deepcopy(general_parameters)
            parameters.update(EmbeddingLSTMParameters(
                embedding_dim=embedding_dim,
                hidden_dim=128,
                sentence_len=sentence_len,
                step_len = 7,
                batch_size=128,
                n_epochs=None,
                n_steps=500000,
                max_ele=50000,
                learning_rate=1e-3,
                freq_threshold=None,
            )._asdict())
            parameters.update(dict(result_path=result_path+"deltalstm_embeddim_"+str(embedding_dim)+
                                   "_sentencelen_"+str(sentence_len)+".csv"))
            print(parameters["result_path"])
            processes.append(mp.Process(
                target=one_experiment,
                args=(EmbeddingLSTM, seq_name, parameters, semaphore, process_id)
            ))
            process_id += 1
            processes[-1].start()
        
    '''
    return


if __name__ == "__main__":
    processes = []
    semaphore = mp.Semaphore(n_processes)
    process_id = 0
    for dataset in datasets:
        experiments_for_one_dataset(dataset, processes, semaphore)
    for p in processes:
        time.sleep(0.5)
        p.join()

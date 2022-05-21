import numpy as np
import math
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ
from QRAM import buildCentroidState, buildVectorsState, encodeVector
from utility import measures

font = {'size'   : 22}

plt.rc('font', **font)

class QKMeans():
    
    """
    QKMeans constructor: 
    
    :param dataset: dataset object
    :param conf: parameters configuration of the algorithm
    """
    def __init__(self, dataset, conf):
        
        self.K = conf['K']
        self.dataset_name = conf['dataset_name']
        self.sc_tresh = conf['sc_tresh']
        self.max_iterations = conf['max_iterations']
        if conf['shots'] is None:
                self.shots = 8192
        else:
            self.shots = conf['shots']
        self.dataset = dataset
        self.data = self.dataset.df
        self.N = self.dataset.N
        self.M = self.dataset.M
        self.centroids = None
        self.old_centroids = None
        
        self.cluster_assignment = [0]*self.M
        
        self.max_qbits = 0
        self.n_circuits = 0
        
        self.ite = 0
        self.SSE_list = []
        self.silhouette_list = []
        self.similarity_list = []
        self.nm_info_list = []
        self.times = []
        self.execution_time_hw = []
        

    """
    computing_cluster: 
    
    Computes the quantum distances between all records and all centroids "sequentially"
        
    :provider (optional, default value=None): real quantum hardware
    """
    def computing_cluster(self, provider=None):
    
        distances = []                         # list of distances: the i-th item is a list of distances between i-th vector and all centroids 
        
        N = self.N

        Rqram_qbits = 1                        # number of qbits for qram register
        Aknn_qbits = 1                         # number of qbits for distance ancilla
        I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
        Aqram_qbits = I_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
        self.max_qbits = I_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits
        #print("total qbits needed:  " + str(Tot_qbits))
        
        a = QuantumRegister(1, 'a')            # ancilla qubit for distance
        i = QuantumRegister(I_qbits, 'i')      # feature index
        r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
        if Aqram_qbits > 0:
            q = QuantumRegister(Aqram_qbits, 'q')  # qram ancilla
    
        outcome = ClassicalRegister(2, 'bit')  # for measuring
        
        tot_execution_time = 0
    
        for index_v, vector in self.dataset.df.iterrows():
            centroid_distances = []            # list of distances between the current vector and all centroids
            for index_c in range(len(self.centroids)):
        
                if Aqram_qbits > 0:
                    circuit = QuantumCircuit(a, i, r, q, outcome)
                else:
                    circuit = QuantumCircuit(a, i, r, outcome)
                    q = None
        
                circuit.h(a)
                circuit.h(i)
    
                #-------------- states preparation  -----------------------#
    
                encodeVector(circuit, self.centroids[index_c], i, a[:]+i[:], r[0], q)  # sample vector encoding in QRAM
    
                circuit.x(a)
    
                encodeVector(circuit, vector, i, a[:]+i[:], r[0], q)    # centroid vector encoding in QRAM
    
                #----------------------------------------------------------#
                circuit.measure(r,outcome[0])
    
                circuit.h(a)
    
                circuit.measure(a,outcome[1])
    
                shots = self.shots
                
                if provider is not None:
                    large_enough_devices = provider.backends(filters=lambda x: x.configuration().n_qubits > 4 and not x.configuration().simulator)
                    backend = least_busy(large_enough_devices)
                    job = execute(circuit, backend, shots=shots)
                    job_monitor(job)
                    result = job.result()
                    execution_time = result.time_taken
                    tot_execution_time += execution_time
                    print("executed in: " + str(execution_time))
                    counts = result.get_counts(circuit)
                else:
                    simulator = Aer.get_backend('qasm_simulator')
                    job = execute(circuit, simulator, shots=shots)
                    result = job.result()
                    counts = result.get_counts(circuit)
                #print("\nTotal counts are:",counts)
                #plot_histogram(counts)
                goodCounts = {k: counts[k] for k in counts.keys() & {'01','11'}}
                #plot_histogram(goodCounts)
                try:
                    n_p0 = goodCounts['01']
                except:
                    n_p0 = 0
                euclidian = 4-4*(n_p0/sum(goodCounts.values()))
                euclidian = math.sqrt(euclidian)

                centroid_distances.append(euclidian)
                
            distances.append(centroid_distances)
        
        self.cluster_assignment = [(i.index(min(i))) for i in distances] # for each vector takes the closest centroid to perform the assignemnt
        
        return tot_execution_time
    
    """
    computing_centroids: 
        
    Computes the new cluster centers as the mean of the records within the same cluster
    """
    def computing_centroids(self):
        #data = data.reset_index(drop=True)
        
        series = []
        for i in range(self.K):
            if i in self.cluster_assignment:
                series.append(self.data.loc[[index for index, n in enumerate(self.cluster_assignment) if n == i]].mean())
            else:
                old_centroid = pd.Series(self.centroids[i])
                old_centroid = old_centroid.rename(lambda x: "f" + str(x))
                series.append(old_centroid)
                
        df_centroids = pd.concat(series, axis=1).T
        self.centroids = self.dataset.normalize(df_centroids).values


    """
    stop_condition: 
        
    Checks if the algorithm have reached the stopping codition
    
    :return: True if the algorithm must terminate, False otherwise
    """
    def stop_condition(self):
        if self.old_centroids is None:
            return False

        with np.errstate(divide='ignore'):
            if np.linalg.norm(self.centroids-self.old_centroids, ord='fro') >= self.sc_tresh:
                return False
        
        return True
    
   
    """
    run_shots: 
        
    Execute the quantum algorithm to check the postselection probabilities
    
    :initial_centroids: vectors chosen as inital centroids
    
    :return: [r1, a0]:
        - r1: postselection probability on the qubit |r>
        - a0: postselection probability on the qubit |a>
    """
    def run_shots(self, initial_centroids):
        self.centroids = initial_centroids
        print("theoretical postselection probability " + str(1/2**(math.ceil(math.log(self.N,2)))))
        r1, a0 = self.computing_cluster_3(check_prob=True)
        print("r1: " + str(r1))
        print("a0: " + str(a0))
        return r1, a0
    

        
    """
    run: 
        
    It executes the algorithm 
    
    :initial_centroids (optional, default_value=None): vectors chosen as initial centroids
    :seed (optional, default value=123): seed to select randomly the initial centroids
    :real_hw (optional, default value=False): if True the algorithm will be executed on real quantum hardware
    """
    def run(self, initial_centroids=None, seed=123, real_hw=False):
        
        if initial_centroids is None:
            self.centroids = self.data.sample(n=self.K, random_state=seed).values
        else:
            self.centroids = initial_centroids
            
        if real_hw:
            provider = IBMQ.load_account()
        else:
            provider = None
            
        #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, initial_space=True)
        #self.dataset.plot2Features(self.data, 'f0', 'f1', self.centroids, cluster_assignment=None, initial_space=True, dataset_name='blobs')
        while not self.stop_condition():
            start = time.time()
            
            self.old_centroids = self.centroids.copy()
            
            print("iteration: " + str(self.ite))
            #print("Computing the distance between all vectors and all centroids and assigning the cluster to the vectors")
            hw_time = self.computing_cluster(provider)
            self.execution_time_hw.append(hw_time)

            #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, True)
    
            #print("Computing new centroids")
            #centroids = computing_centroids_0(data, k)
            self.computing_centroids()
    
            #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, cluster_assignment=self.cluster_assignment, initial_space=False)
            #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, cluster_assignment=self.cluster_assignment, initial_space=True, dataset_name='blobs')
            
            end = time.time()
            elapsed = end - start
            self.times.append(elapsed)
            
            
            # computing measures
            sim = measures.check_similarity(self.data, self.centroids, self.cluster_assignment)
            self.similarity_list.append(sim)
            self.SSE_list.append(self.SSE())
            self.silhouette_list.append(self.silhouette())
            self.nm_info_list.append(self.nm_info())
            
            
            self.ite = self.ite + 1
            
            if self.ite == self.max_iterations:
                break
        
        
    """
    avg_ite_time: 
        
    Returns the average iteration time of the algorithm
    """
    def avg_ite_time(self):
        return round(np.mean(self.times), 2)
    
    """
    avg_ite_hw_time: 
        
    Returns the average iteration time of the algorithm execution on real hardware
    """
    def avg_ite_hw_time(self):
        if len(self.execution_time_hw) > 0:
            return round(np.mean(self.execution_time_hw), 2)
        else:
            return None
    
    """
    avg_sim: 
        
    Returns the average similarity of the cluster assignment produced by the algorithm
    """
    def avg_sim(self):
        return round(np.mean(self.similarity_list), 2)
    
    
    """
    SSE: 
        
    Returns the final Sum of Squared Error
    """
    def SSE(self):
        return round(measures.SSE(self.data, self.centroids, self.cluster_assignment), 3)
    
    
    """
    silhouette: 
        
    Returns the final Silhouette score
    """
    def silhouette(self):
        if len(set(self.cluster_assignment)) <= 1 :
            return None
        else:
            return round(metrics.silhouette_score(self.data, self.cluster_assignment, metric='euclidean'), 3)
    
    """
    vmeasure: 
        
    Returns the final v_measure
    """
    def vmeasure(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.v_measure_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    """
    nm_info: 
        
    Returns the final Normalized Mutual Info Score
    """
    def nm_info(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.normalized_mutual_info_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    """
    save_measure: 
        
    Write into file the measures per iteration
    
    :index: number associated to the algorithm execution
    """
    def save_measures(self, index):
        filename = "result/measures/" + str(self.dataset_name) + "_qkmeans_" + str(index) + ".csv"
        measure_df = pd.DataFrame({'similarity': self.similarity_list, 'SSE': self.SSE_list, 'silhouette': self.silhouette_list, 'nm_info': self.nm_info_list})
        pd.DataFrame(measure_df).to_csv(filename)
      
        
    """
    print_result: 
        
    Write into file the results of the algorithm
    
    :filename (optional, default value=None): name of the file where to save the results
    :process: process number which executed the algorithm 
    :index_conf: configuration number associated to the algorithm execution
    """
    def print_result(self, filename=None, process=0, index_conf=0):
        self.dataset.plotOnCircle(self.data, self.centroids, self.cluster_assignment)
        
        #print("")
        #print("---------------- QKMEANS RESULT ----------------")
        #print("Iterations needed: " + str(self.ite) + "/" + str(self.max_iterations))
        
        avg_time = self.avg_ite_time()
        print("Average iteration time: " + str(avg_time) + " sec")
        
        avg_sim = self.avg_sim()
        print("Average similarity w.r.t classical assignment: " + str(avg_sim) + "%")
        
        SSE = self.SSE()
        print("SSE: " + str(SSE))
        
        silhouette = self.silhouette()
        print("Silhouette score: " + str(silhouette))
        
        vm = self.vmeasure()
        print("Vmeasure: " + str(vm))
        
        nminfo = self.nm_info()
        print("Normalized mutual info score: " + str(nminfo))
    
        fig, ax = plt.subplots()
        ax.plot(range(self.ite), self.similarity_list, marker="o")
        ax.set(xlabel='QKmeans iterations', ylabel='Similarity w.r.t classical assignment')
        ax.set_title("K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", shots = " + str(self.shots))
        #plt.show()
        dt = datetime.datetime.now().replace(microsecond=0)
        #str_dt = str(dt).replace(" ", "_")
        fig.savefig("./plot/qkmeansSim_"+str(process)+"_"+str(index_conf)+".png")
        
        if filename is not None:
            # stampa le cose anche su file 
            
            f = open(filename, 'a')
            f.write("###### TEST " + str(process)+"_"+str(index_conf) + " on " + str(self.dataset_name) + " dataset\n")
            f.write("# Executed on " + str(dt) + "\n")
            f.write("## QKMEANS\n")
            f.write("# Parameters: K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", shots = " + str(self.shots) + "\n")
            f.write("# Iterations needed: " + str(self.ite) + "/" + str(self.max_iterations) + "\n")
            f.write('# Average iteration time: ' + str(avg_time) + 's \n')
            f.write('# Average similarity w.r.t classical assignment: ' + str(avg_sim) + '% \n')
            f.write('# SSE: ' + str(SSE) + '\n')
            f.write('# Silhouette: ' + str(silhouette) + '\n')
            f.write('# Vmeasure: ' + str(vm) + '\n')
            f.write('# Normalized mutual info score: ' + str(nminfo) + '\n')
            f.write("# Quantum kmeans assignment \n")
            f.write(str(self.cluster_assignment))            
            f.write("\n")
            #f.write('# Final centroids \n'
            #self.centroids.to_csv(f, index=False)
            f.close()
       
    """
    print_params: 
        
    Prints the parameters configuration
    """
    def print_params(self, process=0, i=0):
        print("Process " + str(process) + " - configuration: " + str(i) + 
              "\nParameters: K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) +  ", shots = " + str(self.shots) + "\n")


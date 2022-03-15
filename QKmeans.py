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
#from qiskit.circuit.library import MCMT, RYGate
#from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
from QRAM import buildCentroidState, buildVectorsState, encodeVector
from utility import measures

font = {'size'   : 22}

plt.rc('font', **font)

class QKMeans():
    
    def __init__(self, dataset, conf):
        
        self.K = conf['K']
        self.M1 = conf['M1']
        self.dataset_name = conf['dataset_name']
        self.sc_tresh = conf['sc_tresh']
        self.max_iterations = conf['max_iterations']
        if conf['shots'] is None:
            if self.M1 is None:
                self.shots = 8192
            else:
                self.shots = min(self.K * self.M1 * 1024, 500000)
        else:
            self.shots = conf['shots']
        self.quantization = conf['quantization']
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
        
    '''
    Computes the quantum distances between all vectors and all centroids "sequentially": for each vector it computes the 
    quantum euclidian distance with a single centroid for all centroids
    '''
    def computing_cluster_1(self, provider=None):
    
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


    def computing_cluster_2(self, provider=None):
        
        N = self.N
        K = self.K
        
        Aknn_qbits = 1                         # number of qbits for distance ancilla
        I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
        if K == 1:
            C_qbits = 1
        else:
            C_qbits = math.ceil(math.log(K,2))     # number of qbits needed for indexing all centroids
        Rqram_qbits = 1                        # number of qbits for qram register
        Aqram_qbits = I_qbits + C_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
        self.max_qbits = I_qbits + C_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits
        #print("total qbits needed:  " + str(Tot_qbits))
        
        a = QuantumRegister(Aknn_qbits, 'a')   # ancilla qubit for distance
        i = QuantumRegister(I_qbits, 'i')      # feature index
        c = QuantumRegister(C_qbits, 'c')      # cluster
        r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
        if Aqram_qbits > 0:
            q = QuantumRegister(Aqram_qbits, 'q')  # qram ancilla
    
        classical_bit = C_qbits + 2            # C_qbits for clusters + 1 for measuring ancilla + 1 for measuring register    
        outcome = ClassicalRegister(classical_bit, 'bit')  # for measuring
        
        cluster_assignment = []
        
        for index_v, vector in self.dataset.df.iterrows():
            
            if Aqram_qbits > 0:
                circuit = QuantumCircuit(a, i, r, q, c, outcome)
            else:
                circuit = QuantumCircuit(a, i, r, c, outcome)
                q = None
            
            circuit.h(a)
            circuit.h(i)
            circuit.h(c)
            
    
            #--------------------- data vetcor encoding  -----------------------------#
    
            encodeVector(circuit, vector, i, a[:]+i[:], r[0], q)
    
            #-------------------------------------------------------------------------#
    
            circuit.x(a)
            circuit.barrier()
    
            #--------------- centroid vectors encoding -------------------------------#
    
            buildCentroidState(self.centroids, circuit, a, i, c, r, q)
    
            #----------------------------------------------------------------#
            circuit.measure(r, outcome[0])
    
            circuit.h(a) 
    
            circuit.measure(a, outcome[1])
             
            for b in range(C_qbits):
                circuit.measure(c[b], outcome[b+2])
            
            shots = self.shots
            if provider is not None:
                large_enough_devices = provider.backends(filters=lambda x: x.configuration().n_qubits > 4 and not x.configuration().simulator)
                backend = least_busy(large_enough_devices)
                job = execute(circuit, backend, shots=shots)
                job_monitor(job)
                result = job.result()
                counts = result.get_counts(circuit)
            else:
                simulator = Aer.get_backend('qasm_simulator')
                job = execute(circuit, simulator, shots=shots)
                result = job.result()
                counts = result.get_counts(circuit)
            print("\nTotal counts are:",counts)
            #plot_histogram(counts)
            goodCounts = {k: counts[k] for k in counts.keys() if k.endswith('01')} # register 1 and ancilla 0
            cluster = max(goodCounts, key=goodCounts.get)
            print(goodCounts)
            cluster = int(cluster[:C_qbits], 2)
            
            if cluster >= K:
                cluster = 0 
            cluster_assignment.append(cluster)
            print("assigned to cluster: " + str(cluster))
             

        self.cluster_assignment = cluster_assignment 
    
    '''
    Compute all distances between vectors and centroids and assign cluster to every vector in many circuits
    '''
    def computing_cluster_3(self, check_prob=False):
         
        if check_prob:
            r1_list = []
            a0_list = []
        
        N = self.N
        M = self.M
        K = self.K
        M1 = self.M1
        
        Aknn_qbits = 1                         # number of qbits for distance ancilla
        I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
        if K == 1:
            C_qbits = 1
        else:
            C_qbits = math.ceil(math.log(K,2))     # number of qbits needed for indexing all centroids
        Rqram_qbits = 1                        # number of qbits for qram register
        
        a = QuantumRegister(Aknn_qbits, 'a')   # ancilla qubit for distance
        i = QuantumRegister(I_qbits, 'i')      # feature index
        c = QuantumRegister(C_qbits, 'c')      # cluster
        r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
        
        self.n_circuits = math.ceil(M/M1)
        
        #print("circuits needed:  " + str(n_circuits))
        
        cluster_assignment = []
        
        for j in range(self.n_circuits):
            
            #print("Circuit " + str(j+1) + "/" + str(self.n_circuits))
            
            vectors = self.data[j*M1:(j+1)*M1]
            
            if j == self.n_circuits-1:
                M1 = M-M1*(self.n_circuits-1)
                
            #print("vectors to classify: " + str(M1))
            #print("shots: " + str(self.shots))
                
            QRAMINDEX_qbits = math.ceil(math.log(M1,2))     # number of qubits needed to index the qrams (i.e 'test' vectors)
            
            if QRAMINDEX_qbits == 0: # se mi rimane solo un record da assegnare ad un cluster tengo comunque un qubit per qrama anche se non mi serve
                QRAMINDEX_qbits = 1
            
            Aqram_qbits = I_qbits + C_qbits + QRAMINDEX_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
            if C_qbits > QRAMINDEX_qbits:
                Aqram_qbits = Aqram_qbits - QRAMINDEX_qbits
            else:
                Aqram_qbits = Aqram_qbits - C_qbits
            tot_qbits = I_qbits + C_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits + QRAMINDEX_qbits
            if j == 0:
                self.max_qbits = tot_qbits
            #print("total qbits needed for this circuit:  " + str(Tot_qbits))
            
            qramindex = QuantumRegister(QRAMINDEX_qbits,'qramindex')     # index for qrams           
    
            classical_bit = C_qbits + QRAMINDEX_qbits  + 2  # C_qbits for clusters + QRAMINDEX_qbits for qram + 1 for ancilla + 1 for register    
            outcome = ClassicalRegister(classical_bit, 'bit')  # for measuring
                
            if Aqram_qbits > 0:
                q = QuantumRegister(Aqram_qbits, 'q')  # qram ancilla
                circuit = QuantumCircuit(a, i, r, q, c, qramindex, outcome)
            else:
                circuit = QuantumCircuit(a, i, r, c, qramindex, outcome)
                q = None
    
            circuit.h(a)
            circuit.h(i)
            circuit.h(c)
            circuit.h(qramindex)
    
    
            #--------------------- data vetcor encoding  -----------------------------#
    
            buildVectorsState(vectors, circuit, a, i, qramindex, r, q)
    
            #-------------------------------------------------------------------------#
    
            circuit.x(a)
            circuit.barrier()
    
            #--------------- centroid vectors encoding -------------------------------#
    
            buildCentroidState(self.centroids, circuit, a, i, c, r, q)
    
            #----------------------------------------------------------------#
    
    
            circuit.measure(r, outcome[0])
    
            circuit.h(a) 
    
            #return circuit
    
            circuit.measure(a, outcome[1])
    
            # measuring cluster bits
            for b in range(C_qbits):
                circuit.measure(c[b], outcome[b+2])
    
            # measuring qram bits
            for b in range(QRAMINDEX_qbits):
                circuit.measure(qramindex[b], outcome[b+2+C_qbits])
    
            simulator = Aer.get_backend('qasm_simulator')
            job = execute(circuit, simulator, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            #print("\nTotal counts are:",counts)
            #plot_histogram(counts)
            # qram-classe-ancilla-registro
            
            if check_prob:
                r1 = {k: counts[k] for k in counts.keys() if k.endswith('1')}
                r1_perc = (sum(r1.values())/self.shots)*100
                r1_list.append(r1_perc)
                
                a0 = {k: r1[k] for k in r1.keys() if k.endswith('01')}
                a0_perc = (sum(a0.values())/(sum(r1.values()))*100)
                a0_list.append(a0_perc)
            
            
            goodCounts = {k: counts[k] for k in counts.keys() if k.endswith('01')} 
            #plot_histogram(goodCounts)
    
            for v in range(M1):
                strformat = "{0:0" + str(QRAMINDEX_qbits) + "b}"
                strbin = strformat.format(v)
                vi_counts = {k: goodCounts[k] for k in goodCounts.keys() if k.startswith(strbin)} 
    
                if len(vi_counts) == 0:
                    cluster = 0
                else:
                    count_sorted = sorted(((v,k) for k,v in vi_counts.items()))
                    cluster = count_sorted[-1][1] # is the key related to the maximum count
                    #cluster = max(vi_counts, key=vi_counts.get)
                    cluster = int(cluster[QRAMINDEX_qbits:-2],2)
    
                #print('cluster prima ' + str(cluster))
                ## SEMPLICE EURISTICA: QUANDO RESTITUISCO UN CLUSTER CHE NON ESISTE ALLORA PRENDO IL SECONDO PIU ALTO 
                ind = 2
                while cluster >= K:
                    if ind > len(count_sorted):
                        break
                    cluster = count_sorted[-ind][1]
                    cluster = int(cluster[QRAMINDEX_qbits:-2], 2)
                    ind = ind + 1
                if cluster >= K:
                    cluster = 0
    
                #print('cluster dopo ' + str(cluster))
                #print('------------------------------------------')
    
                cluster_assignment.append(cluster)
                #print("vector " + str(i))
                #print("Quantum assignment: " + str(cluster))    
    
            #df.loc[j*M1:(j+1)*M1-1,'cluster'] = cluster_assignment 
        self.cluster_assignment = cluster_assignment
        
        if check_prob:
            return round(sum(r1_list)/len(r1_list),3),round(sum(a0_list)/len(a0_list),3)
        
            
    
    '''
    Computes the new cluster centers as the mean of the records within the same cluster
    '''
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


    def stop_condition(self):
        if self.old_centroids is None:
            return False

        '''
        for i in range(len(self.centroids)):
            difference = np.linalg.norm(self.centroids[i]-self.old_centroids[i])
            if difference > self.sc_tresh:
                return False
        '''
        if np.linalg.norm(self.centroids-self.old_centroids, ord='fro') >= self.sc_tresh:
            return False
        
        return True
    
    
    def run_shots(self, initial_centroids):
        self.centroids = initial_centroids
        print("theoretical postselection probability " + str(1/2**(math.ceil(math.log(self.N,2)))))
        '''
        counts = self.computing_cluster(check_prob=True)
        r1 = {k: counts[k] for k in counts.keys() if k.endswith('1')}
        r0 = {k: counts[k] for k in counts.keys() if k.endswith('0')}
        print("r1: " + str((sum(r1.values())/self.shots)*100))
        print("r0: " + str((sum(r0.values())/self.shots)*100))
        

        a0 = {k: r1[k] for k in r1.keys() if k.endswith('01')}
        a1 = {k: r1[k] for k in r1.keys() if k.endswith('11')}
        print("a0: " + str((sum(a0.values())/(sum(r1.values()))*100)))
        print("a1: " + str((sum(a1.values())/(sum(r1.values()))*100)))
        return str((sum(r1.values())/self.shots)*100)  
        '''
        r1, a0 = self.computing_cluster_3(check_prob=True)
        print("r1: " + str(r1))
        print("a0: " + str(a0))
        return r1, a0
    

        
    
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
            if self.quantization == 1:
                self.computing_cluster_1(provider)
            elif self.quantization == 2:
                self.computing_cluster_2(provider)
            else:
                self.computing_cluster_3()
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
        
        
        
    def avg_ite_time(self):
        return round(np.mean(self.times), 2)
    
    def avg_sim(self):
        return round(np.mean(self.similarity_list), 2)
    
    def SSE(self):
        return round(measures.SSE(self.data, self.centroids, self.cluster_assignment), 3)
    
    def silhouette(self):
        if len(set(self.cluster_assignment)) <= 1 :
            return None
        else:
            return round(metrics.silhouette_score(self.data, self.cluster_assignment, metric='euclidean'), 3)
    
    def vmeasure(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.v_measure_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    def nm_info(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.normalized_mutual_info_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    def save_measures(self, index):
        filename = "result/measures/" + str(self.dataset_name) + "_qkmeans_" + str(index) + ".csv"
        measure_df = pd.DataFrame({'similarity': self.similarity_list, 'SSE': self.SSE_list, 'silhouette': self.silhouette_list, 'nm_info': self.nm_info_list})
        pd.DataFrame(measure_df).to_csv(filename)
        
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
        ax.set_title("K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", M1 = " + str(self.M1) + ", shots = " + str(self.shots))
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
            f.write("# Parameters: K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", M1 = " + str(self.M1) + ", shots = " + str(self.shots) + "\n")
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
            
    def print_params(self, process=0, i=0):
        print("Process " + str(process) + " - configuration: " + str(i) + 
              "\nParameters: K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", M1 = " + str(self.M1) + ", shots = " + str(self.shots) + "\n")


'''
if __name__ == "__main__":

    dataset = Dataset('noisymoon')
    conf = {
            "dataset_name": 'noisymoon',
            "K": 3,
            "M1": 10,
            "sc_tresh": 0,
            "max_iterations": 5
        }
    
    QKMEANS = QKMeans(dataset, conf, 321)
    QKMEANS.print_params()
    QKMEANS.run()
    QKMEANS.print_result("log/log.txt")
'''
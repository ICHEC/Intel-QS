//------------------------------------------------------------------------------
// Copyright (C) 2017 Intel Corporation 
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

#include "../include/qureg.hpp"

// The scope is applying a sequence of num_gates=40 gates to the quantum register.
// The form of each gate is the same:
//     controlled 1-qubit operation defined by the 2x2 matrix G
// but each pair (control,target) for the involved qubit is randomly generated.

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    unsigned myrank=0, nprocs=1;
    qhipster::mpi::Environment env(argc, argv);
    myrank = env.GetStateRank();
    nprocs = qhipster::mpi::Environment::GetStateSize();
    if (env.IsUsefulRank() == false) return 0;
    int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
            num_threads = omp_get_num_threads();
    }
#endif

/// --- PARAMETERS ------------------------------------------- ///
    // number of qubits
    unsigned num_qubits_compute = 8;
    // number of (two-qubit) gates
    unsigned num_gates=20;
    // number of repetition of the (stochastic) noisy circuit
    unsigned num_noisy_circuits=num_gates*10;
    // T_1 and T_2 times for slow decoherence
    double T_1_slow=1000. , T_2_slow=500. ;
    double T_1_fast=40.     , T_2_fast=20.    ;
    // T_1 and T_2 times for slow decoherence
/// ---------------------------------------------------------- ///

    std::size_t tmp_size = 0;
    if(argc != 2)
    {
            fprintf(stderr, "usage: %s <num_qubits_compute> \n", argv[0]);
            exit(1);
    }
    else
    {
            num_qubits_compute = atoi(argv[1]);
    }

    // simplified pauliX gate: Pauli X
    TM2x2<ComplexDP> pauliX;
    pauliX(0, 0) = {0. , 0.};
    pauliX(0, 1) = {1. , 0.};
    pauliX(1, 0) = {1. , 0.};
    pauliX(1, 1) = {0. , 0.};

    QubitRegister<ComplexDP> psi0(num_qubits_compute);
    psi0.Initialize("base", 0);

    // Setup compute and auxiliary quantum register.
    std::size_t num_qubits_auxiliary = 2*num_qubits_compute + 2;
    std::vector<std::size_t> reg_compute(num_qubits_compute);
    std::vector<std::size_t> reg_auxiliary(num_qubits_auxiliary);
    std::size_t qubit_index = 0;
    for(std::size_t i = 0; i < num_qubits_compute; i++){
        reg_compute[i] = qubit_index;
        qubit_index++;
    }
    for(std::size_t i = 0; i < num_qubits_auxiliary; i++){
        reg_auxiliary[i] = qubit_index;
        qubit_index++;
    }

    {
        psi0.EnableStatistics();
   
        for(int qubit_id = 0; qubit_id < num_qubits_compute; qubit_id++)
            psi0.Apply1QubitGate(reg_compute[qubit_id], pauliX);

        psi0.Print("Before NCU");
        psi0.GetStatistics();

        std::size_t num_control_qubits = num_qubits_compute - 1;
        std::vector<std::size_t> control_ids(num_control_qubits);
        for(int i = 0; i < num_control_qubits; i++){
            control_ids[i] = reg_compute[i];
        }
        std::size_t target_id = num_qubits_compute - 1;

        psi0.ApplyNCU(pauliX, control_ids, reg_auxiliary, target_id);
        psi0.Print("After NCU");

    }

 
// ---------------- slow decoherence
// 
/*
    if (myrank==0) std::cout << " slow decoherence \n";
    double over_sq_1 = 0.;
    for (unsigned j=0; j<num_noisy_circuits; j++)
    {
            psi1.Initialize("base", 0);
            psi1.ResetTimeForAllQubits();
            for (auto &p : qpair)
                    psi1.ApplyControlled1QubitGate(p.first, p.second, G);
        
            for(int pos = 0; pos < num_qubits_compute; pos++)
                    psi1.Apply1QubitGate(pos, G);

            psi1.ApplyNoiseGatesOnAllQubits();
            over_sq_1 = over_sq_1 + std::norm( psi0.ComputeOverlap(psi1) ) ;
            if (j%100==0 && myrank==0)
                    std::cout << " j=" << j << " , logical gate count for"
                                        << " psi1 =" << psi1.GetTotalExperimentalGateCount()
                                        << " (they should be " << num_gates+num_qubits_compute << ") \n";
    }
    over_sq_1 = over_sq_1/(double)num_noisy_circuits;


    // ---------------- fast decoherence
    if (myrank==0) std::cout << " fast decoherence \n";
    double over_sq_2 = 0.;
    for (unsigned j=0; j<num_noisy_circuits; j++)
    {
            psi2.Initialize("base", 0);
            psi2.ResetTimeForAllQubits();
            for (auto &p : qpair)
                    psi2.ApplyControlled1QubitGate(p.first, p.second, G);
        
            for (int pos = 0; pos < num_qubits_compute; pos++)
                    psi2.Apply1QubitGate(pos, G);

            psi2.ApplyNoiseGatesOnAllQubits();
            over_sq_2 = over_sq_2 + std::norm( psi0.ComputeOverlap(psi2) ) ;
    }
    over_sq_2 = over_sq_2/(double)num_noisy_circuits;
    // ---------------- 
    // computation of the overlap between the ideal state and those exposed to noise
    if (myrank == 0) 
            std::cout << " Overlap-squared between ideal and 'slow decoherence' state = "
                                << over_sq_1 << "\n"
                                << " Overlap-squared between ideal and 'fast decoherence' state = "
                                << over_sq_2 << "\n";

*/

//    double e = psi2.MaxAbsDiff(psi1);
}

/**
 * @file ncu.hpp
 * @author Lee J. O'Riordan (lee.oriordan@ichec.ie)
 * @author Myles Doyle (myles.doyle@ichec.ie)
 * @brief Application to show functionality of ApplyNCU gate call.
 * @version 0.2
 * @date 2020-06-12
 */

#include "../include/qureg.hpp"

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

    // number of qubits in compute register.
    std::size_t num_qubits_compute = 8;
    if(argc != 2){
        fprintf(stderr, "usage: %s <num_qubits_compute> \n", argv[0]);
        exit(1);
    }
    else{
        num_qubits_compute = atoi(argv[1]);
    }

    // pauliX gate that will be applied in NCU
    TM2x2<ComplexDP> pauliX;
    pauliX(0, 0) = {0. , 0.};
    pauliX(0, 1) = {1. , 0.};
    pauliX(1, 0) = {1. , 0.};
    pauliX(1, 1) = {0. , 0.};

    QubitRegister<ComplexDP> psi(num_qubits_compute);
    psi.Initialize("base", 0);

    // Setup vector to store compute and auxiliary quantum register indices.
    // |compute reg>|auxiliary reg>
    std::size_t num_qubits_auxiliary = num_qubits_compute;
    std::vector<std::size_t> reg_compute(num_qubits_compute);
    std::vector<std::size_t> reg_auxiliary(num_qubits_auxiliary);
    std::size_t qubit_index = 0;

    // Set qubit indices of registers
    for(std::size_t i = 0; i < num_qubits_compute; i++){
        reg_compute[i] = qubit_index;
        qubit_index++;
    }
    for(std::size_t i = 0; i < num_qubits_auxiliary; i++){
        reg_auxiliary[i] = qubit_index;
        qubit_index++;
    }


    {
        psi.EnableStatistics();
   
        // Apply a Hadamard gate to first num_qubits_compute-1
        // qubits in the compute register.
        for(int qubit_id = 0; qubit_id < num_qubits_compute-1; qubit_id++){
            //psi.Apply1QubitGate(reg_compute[qubit_id], pauliX);
            psi.ApplyHadamard(reg_compute[qubit_id]);
        }

        psi.Print("Before NCU");
        psi.GetStatistics();

        // Set qubit indices for qubits acting as control
        std::size_t num_control_qubits = num_qubits_compute - 1;
        std::vector<std::size_t> control_ids(num_control_qubits);

        // Set vector containing indices of the qubits acting as
        // control for the NCU gate.
        for(int std::size_t = 0; i < num_control_qubits; i++){
            control_ids[i] = reg_compute[i];
        }

        // Set index of target qubit
        std::size_t target_id = num_qubits_compute - 1;

        // Apply NCU
        psi.ApplyNCU(pauliX, control_ids, reg_auxiliary, target_id);

        // Observe only state with the first num_qubits_compute-1 
        // qubits in the compute register set to 1 executes PauliX
        // on the target qubit.
        psi.Print("After NCU");
    }
}

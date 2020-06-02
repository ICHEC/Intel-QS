/**
 * @file ncu.hpp
 * @author Lee J. O'Riordan (lee.oriordan@ichec.ie)
 * @author Myles Doyle (myles.doyle@ichec.ie)
 * @brief Functions for applying n-qubit controlled U (unitary) gates. Adapted from QNLP.
 * @version 0.2
 * @date 2020-06-01
 */

#ifndef QNLP_NCU
#define QNLP_NCU

#include <unordered_map>
#include <complex>
#include <cassert>
#include <utility>
#include <vector>
#include <iostream>

#include <GateCache.hpp>
#include <mat_ops.hpp>

/**
 * @brief Class definition for applying n-qubit controlled unitary operations.
 * 
 * @tparam SimulatorType Class Simulator Type
 */
template <class Matrix2x2Type>
class NCU{
    private:

    std::unordered_set<std::string> default_gates {"X", "Y", "Z", "I", "H"};
    GateCache<Matrix2x2Type> gate_cache;

    protected:
    static std::size_t num_gate_ops;

    public:
    /**
     * @brief Construct a new NCU object
     * 
     */
    NCU() {
        gate_cache = GateCache<SimulatorType>(qSim);
    };

    /**
     * @brief Destroy the NCU object
     * 
     */
    ~NCU(){
        clearMaps();
        gate_cache.clearCache();
    };

    /**
     * @brief Add the PauliX and the given unitary U to the maps
     * 
     * @param U 
     */
    void initialiseMaps( std::size_t num_ctrl_lines){
        gate_cache.initCache(num_ctrl_lines);
    }

    /**
     * @brief Add the given unitary matrix to the maps up to the required depth
     * 
     * @param U 
     */
    void addToMaps( std::string U_label, const Mat2x2Type& U, std::size_t num_ctrl_lines){
        gate_cache.addToCache( U_label, U, num_ctrl_lines);
    }

    /**
     * @brief Get the Map of cached gates. Keys are strings, and values are vectors of paired (gate, gate adjoint) types where the index give the value of (gate)^(1/2^i)
     * 
     * @return GateCache<SimulatorType> type 
     */
    GateCache<Matrix2x2Type>& getGateCache(){
        return gate_cache;
    }

    /**
     * @brief Clears the maps of stored sqrt matrices
     * 
     */
    void clearMaps(){
        gate_cache.clearCache();
    }

    /**
     * @brief Decompose n-qubit controlled op into 1 and 2 qubit gates. Control indices can be in any specified location. Ensure the gate cache has been populated with the appropriate gate type before running. This avoids O(n) checking of the container at each call for the associated gates.
     * 
     * @tparam Type ComplexDP or ComplexSP 
     * @param qReg Qubit register
     * @param ctrlIndices Vector of indices for control lines
     * @param qTarget Target qubit for the unitary matrix U
     * @param U Unitary matrix, U
     * @param depth Depth of recursion.
     */
    void applyNQubitControl(SimulatorType& qSim, 
            const std::vector<std::size_t> ctrlIndices,
            const std::vector<std::size_t> auxIndices,
            const unsigned int qTarget,
            const std::string gateLabel,
            const Mat2x2Type& U,
            const std::size_t depth
    ){
        int local_depth = depth + 1;

        //Determine the range over which the qubits exist; consider as a count of the control ops, hence +1 since extremeties included
        std::size_t cOps = ctrlIndices.size();

        // Assuming given a set of auxiliary qubits, utilise the qubits for better depth optimisation.
        if( (cOps >= 5) && ( auxIndices.size() >= cOps-2 ) && (gateLabel == "X") && (depth == 0) ){ //161 -> 60 2-qubit gate calls
            qSim.applyGateCCX( ctrlIndices.back(), *(auxIndices.begin() + ctrlIndices.size() - 3), qTarget);

            for (std::size_t i = ctrlIndices.size()-2; i >= 2; i--){
                qSim.applyGateCCX( *(ctrlIndices.begin()+i), *(auxIndices.begin() + (i-2)), *(auxIndices.begin() + (i-1)));
            }
            qSim.applyGateCCX( *(ctrlIndices.begin()), *(ctrlIndices.begin()+1), *(auxIndices.begin()) );
            
            for (std::size_t i = 2; i <= ctrlIndices.size()-2; i++){
                qSim.applyGateCCX( *(ctrlIndices.begin()+i), *(auxIndices.begin()+(i-2)), *(auxIndices.begin()+(i-1)));
            }
            qSim.applyGateCCX( ctrlIndices.back(), *(auxIndices.begin() + ctrlIndices.size() - 3), qTarget);

            for (std::size_t i = ctrlIndices.size()-2; i >= 2; i--){
                qSim.applyGateCCX( *(ctrlIndices.begin()+i), *(auxIndices.begin() + (i-2)), *(auxIndices.begin() + (i-1)));
            }
            qSim.applyGateCCX( *(ctrlIndices.begin()), *(ctrlIndices.begin()+1), *(auxIndices.begin()) );
            for (std::size_t i = 2; i <= ctrlIndices.size()-2; i++){
                qSim.applyGateCCX( *(ctrlIndices.begin()+i), *(auxIndices.begin()+(i-2)), *(auxIndices.begin()+(i-1)));
            }
        }
        // Optimisation for replacing 17 2-qubit with 13 2-qubit gate calls
        else if(cOps == 3){ 
            //Apply the 13 2-qubit gate calls
            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth+1].first, ctrlIndices[0], qTarget, gateLabel );
            qSim.applyGateCX( ctrlIndices[0], ctrlIndices[1]);

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth+1].second, ctrlIndices[1], qTarget, gateLabel );
            qSim.applyGateCX( ctrlIndices[0], ctrlIndices[1]);

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth+1].first, ctrlIndices[1], qTarget, gateLabel );
            qSim.applyGateCX( ctrlIndices[1], ctrlIndices[2]);

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth+1].second, ctrlIndices[2], qTarget, gateLabel );
            qSim.applyGateCX( ctrlIndices[0], ctrlIndices[2]);

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth+1].first, ctrlIndices[2], qTarget, gateLabel );
            qSim.applyGateCX( ctrlIndices[1], ctrlIndices[2]);

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth+1].second, ctrlIndices[2], qTarget, gateLabel );
            qSim.applyGateCX( ctrlIndices[0], ctrlIndices[2]);

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth+1].first, ctrlIndices[2], qTarget, gateLabel );
        }
        // Default quadratic decomposition from Barenco et al (1995)
        else if (cOps >= 2 && cOps !=3){
            std::vector<std::size_t> subCtrlIndices(ctrlIndices.begin(), ctrlIndices.end()-1);

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth].first, ctrlIndices.back(), qTarget, gateLabel );

            applyNQubitControl(qSim, subCtrlIndices, auxIndices, ctrlIndices.back(), "X", qSim.getGateX(), 0 );

            qSim.applyGateCU( gate_cache.gateCacheMap[gateLabel][local_depth].second, ctrlIndices.back(), qTarget, gateLabel );

            applyNQubitControl(qSim, subCtrlIndices, auxIndices, ctrlIndices.back(), "X", qSim.getGateX(), 0 );

            applyNQubitControl(qSim, subCtrlIndices, auxIndices, qTarget, gateLabel, gate_cache.gateCacheMap[gateLabel][local_depth+1].first, local_depth );
        }

        // If the number of control qubits is less than 2, assume we have decomposed sufficiently
        else{
            qSim.applyGateCU(gate_cache.gateCacheMap[gateLabel][depth].first, ctrlIndices[0], qTarget, gateLabel);
        }
    }
};
#endif
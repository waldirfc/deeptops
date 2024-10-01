# Install script for directory: ${CMAKE_CURRENT_SOURCE_DIR}/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/static" TYPE STATIC_LIBRARY FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/libToPS.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tops" TYPE FILE FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AkaikeInformationCriteria.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Alphabet.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BayesianInformationCriteria.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ConfigurationReader.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ContextTree.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/DecodableModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/DegenerateDistribution.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/FactorableModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/FactorableModelPrefixSumArray.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/DiscreteIIDModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/DiscreteIIDModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/FixedSequenceAtPosition.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/FixedSequenceAtPositionCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GHMMStates.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GeneralizedHiddenMarkovModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GeneralizedHiddenMarkovModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/HiddenMarkovModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/HiddenMarkovModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/InhomogeneousFactorableModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/InhomogeneousMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/InhomogeneousMarkovChainCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/MultipleAlignment.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/NullPrefixSumArray.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PhasedFactorableModelEvaluationAlgorithm.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PhasedRunLengthDistribution.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PhasedRunLengthDistributionCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PrefixSumArray.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProbabilisticModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProbabilisticModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProbabilisticModelCreatorClient.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProbabilisticModelDecorator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProbabilisticModelParameter.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/RemoveSequenceFromModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ReverseComplementDNA.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ReverseComplementDNACreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Sequence.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SequenceEntry.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SequenceFactory.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SequenceFormat.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SmoothedHistogramBurge.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SmoothedHistogramKernelDensity.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SmoothedHistogramStanke.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SparseMatrix.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Symbol.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TargetModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TargetModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainDiscreteIIDModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainFixedLengthMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainInterpolatedMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainHMMBaumWelch.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainNeuralNetwork.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainHMMMaximumLikelihood.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainPhasedMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainInterpolatedPhasedMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainPhasedMarkovChainContextAlgorithm.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainVariableLengthInhomogeneousMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainVariableLengthMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainWeightArrayModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/VariableLengthMarkovChain.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/VariableLengthMarkovChainCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/util.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainGHMMTransitions.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BernoulliModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SimilarityBasedSequenceWeighting.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SimilarityBasedSequenceWeightingCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainSimilarityBasedSequenceWeighting.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/MultipleSequentialModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/MultipleSequentialModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/StoreLoadedModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PairHiddenMarkovModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PairHiddenMarkovModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainPHMMBaumWelch.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/MaximumDependenceDecomposition.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProfileHiddenMarkovModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ProfileHiddenMarkovModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainProfileHMMMaxLikelihood.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TrainProfileHMMBaumWelch.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/NeuralNetworkModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/NeuralNetworkModelCreator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/OverlappedProbabilisticModel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/OverlappedProbabilisticModelCreator.hpp"
    )
endif()


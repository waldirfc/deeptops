/*
 *       NeuralNetworkModel.cpp
 *
 *       Copyright 2011 Waldir Caro <waldirc@ime.usp.br>
 *                      Alan Mitchell Durham <aland@usp.br>
 *
 *       This program is free software; you can redistribute it and/or modify
 *       it under the terms of the GNU  General Public License as published by
 *       the Free Software Foundation; either version 3 of the License, or
 *       (at your option) any later version.
 *
 *       This program is distributed in the hope that it will be useful,
 *       but WITHOUT ANY WARRANTY; without even the implied warranty of
 *       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *       GNU General Public License for more details.
 *
 *       You should have received a copy of the GNU General Public License
 *       along with this program; if not, write to the Free Software
 *       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *       MA 02110-1301, USA.
 */

#include "NeuralNetworkModel.hpp"
#include "NeuralNetworkModelCreator.hpp"
#include "Symbol.hpp"
#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include <iterator>

namespace tops {

    NeuralNetworkModel::NeuralNetworkModel(){
        _trained_model_file = "";
    }

    NeuralNetworkModel::NeuralNetworkModel(std::shared_ptr<torch::nn::Sequential> module_nn_ptr){
        _module_nn = *module_nn_ptr;
        _trained_model_file = "";
    }

    NeuralNetworkModel::NeuralNetworkModel(std::string trained_model_file){
        _trained_model_file = trained_model_file;
    }

    // ToDo: UPDATE to be the same as the config file.
    std::string NeuralNetworkModel::str () const {
        std::stringstream out;
        out << "model_name = \"NeuralNetworkModel\"\n" ;
        out << "layers = (\n";
        
        //std::cout << "layers size: " << _module_nn->size() << std::endl;

        for(size_t i = 0; i < _module_nn->size(); ++i) {
            const std::shared_ptr<torch::nn::Module>& module_ptr = _module_nn->ptr(i);            
            module_ptr->pretty_print(out);
            out << "\n";
        }
        out << ")\n";

        out << "sequence length = \"" << _sequence_length << "\"\n";
        out << "trained_model_file = \"" << _trained_model_file << "\"\n";        
        
        return out.str();
    }

    void NeuralNetworkModel::setParameters(std::shared_ptr<torch::nn::Sequential> module_nn_ptr, std::string trained_model_file, int sequence_length) {
        _module_nn = *module_nn_ptr;
        _trained_model_file = trained_model_file;
        _sequence_length = sequence_length;

        if (_trained_model_file != ""){ //there is a trained model in jit format
            try {
                // Deserialize the ScriptModule from a file using torch::jit::load()
                _trained_module_nn = torch::jit::load(_trained_model_file);
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading the trained model\n";
            }
        }
        //std::cout << this->str();
    }

    ProbabilisticModelCreatorPtr NeuralNetworkModel::getFactory () const{
        return NeuralNetworkModelCreatorPtr(new NeuralNetworkModelCreator());
    }

    void NeuralNetworkModel::initialize(const ProbabilisticModelParameters & p) {
        //ProbabilisticModelParameterValuePtr weight = p.getMandatoryParameterValue("weight");
        //ProbabilisticModelParameterValuePtr bias = p.getMandatoryParameterValue("bias");
        ProbabilisticModelParameterValuePtr module_nn_ptr = p.getMandatoryParameterValue("layers");
        ProbabilisticModelParameterValuePtr sequence_length_p = p.getMandatoryParameterValue("sequence_length");
        ProbabilisticModelParameterValuePtr model_file = p.getOptionalParameterValue("trained_model_file");
        ProbabilisticModelParameterValuePtr symbols = p.getOptionalParameterValue("alphabet");

        // load a trained model if it is present                
        std::shared_ptr<torch::nn::Sequential> module_nn = module_nn_ptr->getModule();
        
        int sequence_length = sequence_length_p->getInt();
        std::string trained_model_file = "";
        if (model_file != NULL) trained_model_file = model_file->getString();
        
        AlphabetPtr alphabet = AlphabetPtr(new Alphabet());
        alphabet->initializeFromVector(symbols->getStringVector());
        setAlphabet(alphabet);

        setParameters(module_nn, trained_model_file, sequence_length);
    }

    ProbabilisticModelParameters NeuralNetworkModel::parameters () const {
        
        ProbabilisticModelParameters par;
        par.add("model_name", StringParameterValuePtr(new StringParameterValue("NeuralNetworkModel")));
        auto module_nn = std::make_shared<torch::nn::Sequential>(_module_nn);
        par.add("layers", ModuleParameterValuePtr(new ModuleParameterValue(module_nn)));
        par.add("sequence_length", IntParameterValuePtr(new IntParameterValue(_sequence_length)));
        if (_trained_model_file != "") par.add("trained_model_file", StringParameterValuePtr(new StringParameterValue(_trained_model_file)));
        return par;
    }

    // Transform input_data (vector<vector<int>>) in tensor_data (Tensor)
    torch::Tensor NeuralNetworkModel::sequences_to_Tensor(SequenceList & input_data) const {
        
        // Step 1: Convert the vector of vectors to a flat vector
        std::vector<int> flat_data;
        for (const auto& row : input_data) {
            flat_data.insert(flat_data.end(), row.begin(), row.end());
        }

        // Step 2: Convert the flat vector to a torch::Tensor
        torch::Tensor tensor_data = torch::from_blob(flat_data.data(), {input_data.size(), input_data[0].size()}, torch::kInt);

        // Step 3: Find the maximum value in the tensor to determine the number of classes
        int num_classes = tensor_data.max().item<int>() + 1; // Assuming classes are zero-indexed

        // Step 4: Convert the tensor to a one-hot encoded tensor
        torch::Tensor one_hot_tensor = torch::one_hot(tensor_data, num_classes);

        // Step 5: Convert the one-hot tensor to the desired data type (e.g., float)
        one_hot_tensor = one_hot_tensor.to(torch::kFloat);

        return one_hot_tensor;
    }

    // forward method using the sequential architecture neural network
    double NeuralNetworkModel::trainSGDAlgorithm(SequenceList & training_set, int epochs, int batch_size, double learning_rate) {
        
        // Hyperparameters
        double loss_value = HUGE;
        // int batch_size = 64;
        // double learning_rate = 0.01;

        // Transform the training set into Tensors
        torch::Tensor input = sequences_to_Tensor(training_set);

        // Specify the batch size
        int num_batches = input.size(0) / batch_size;        
              
        // Define the loss function and optimizer
        torch::nn::MSELoss loss_fn;
        // Instantiate an SGD optimization algorithm to update our net's parameters.
        torch::optim::SGD optimizer(_module_nn->parameters(), torch::optim::SGDOptions(learning_rate));


        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            size_t batch_index = 0;
            // Iterate the input data to yield batches from the dataset.
            for (int i = 0; i < num_batches; ++i) {

                // Reset gradients.
                //optimizer.zero_grad();
                _module_nn->zero_grad();

                // Get i batch data
                torch::Tensor batch_data = input.slice(0, i * batch_size, (i + 1) * batch_size);
                
                // Execute the model (forward) on the input batch data.
                torch::Tensor predictions = _module_nn->forward(batch_data);

                // Compute a loss value to judge the prediction of our model.
                torch::Tensor loss = loss_fn(predictions, /*batch_data.target*/batch_data.slice(1, batch_data.size(1) - 1, batch_data.size(1)));
                
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                
                // Update the parameters based on the calculated gradients.
                optimizer.step();
                
                // Output the loss and checkpoint every 10 batches.
                if (++batch_index % 100 == 0) {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                    << " | Loss: " << loss.item<float>() << std::endl;
                }
                loss_value = loss.item<double>();
            }
        }
        
        return loss_value;
    }

    // \@begin should be 0 and @end should be the same as sequence_length
    double NeuralNetworkModel::evaluate(const Sequence & s, unsigned int begin, unsigned int end)  {
        std::cout << "seq size: " << s.size() << std::endl;
        if(s.size() == _sequence_length){
            //std::vector<std::vector<int> > input;
            //input.push_back(s);
            //torch::Tensor input_t = sequences_to_Tensor(input);
            // Step 1: Extract the subsequence
            Sequence subsequence(s.begin() + begin, s.begin() + end);

            // Step 2: Create a SequenceList from the subsequence
            SequenceList sample = {subsequence};

            // Step 3: Convert the SequenceList to a torch::Tensor using sequences_to_Tensor
            // As sequences_to_Tensor is a non-const function, cast away the constness
            //auto non_const_this = const_cast<NeuralNetworkModel*>(this);
            //torch::Tensor input_t = non_const_this->sequences_to_Tensor(sample);
            torch::Tensor input_t = sequences_to_Tensor(sample);

            // Step 4: Pass the tensor to the model's forward function
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_t);

            // Make sure the module is not null and has been loaded correctly
            assert(_trained_module_nn);
            //auto non_const_this_module = const_cast<torch::jit::Module*>(_trained_module_nn);

            std::cout << "one-hot: " << input_t.sizes() << std::endl;
            torch::Tensor output = _trained_module_nn.forward({input_t}).toTensor();
            // Convert each element to double
            double value1 = output[0].item<double>();
            double value2 = output[1].item<double>();
            if(value1 > value2) return value1;
            else return value2;
        }
        else {
            std::cerr << "Invalid length sequence" << std::endl;
            return 0.0;
        }
    }

    double NeuralNetworkModel::choose() const {
        //std::cerr << "Not implemented" << std::endl;
        return 0.0;
    }

    Sequence & NeuralNetworkModel::choose(Sequence & s, int size ) const {
        //std::cerr << "Not implemented" << std::endl;
        return s;
    }

}



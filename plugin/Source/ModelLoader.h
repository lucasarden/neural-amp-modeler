/*
  ==============================================================================

    ModelLoader.h
    Load model weights and architecture from JSON file

  ==============================================================================
*/

#pragma once

#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <vector>

struct ModelArchitecture
{
    int channels;
    int numLayers;
    int kernelSize;
    int dilationBase;
    std::vector<int> dilations;
    int receptiveField;
    float latencyMs;
    int sampleRate;
};

struct LayerWeights
{
    std::vector<std::vector<std::vector<float>>> convWeight; // [out_ch][in_ch][kernel]
    std::vector<float> convBias;
    int dilation;
};

struct ModelWeights
{
    // Input layer (1 -> channels)
    std::vector<float> inputWeight; // [channels]
    std::vector<float> inputBias;   // [channels]

    // Residual blocks
    std::vector<LayerWeights> residualBlocks;

    // Output layer (channels -> 1)
    std::vector<float> outputWeight; // [channels]
    std::vector<float> outputBias;   // [1]
};

class ModelLoader
{
public:
    ModelLoader() = default;

    bool loadFromFile(const juce::File& jsonFile)
    {
        if (!jsonFile.existsAsFile())
        {
            DBG("Model file does not exist: " << jsonFile.getFullPathName());
            return false;
        }

        juce::String jsonText = jsonFile.loadFileAsString();
        auto json = juce::JSON::parse(jsonText);

        if (!json.isObject())
        {
            DBG("Failed to parse JSON");
            return false;
        }

        auto root = json.getDynamicObject();
        if (root == nullptr)
            return false;

        // Load architecture
        if (!loadArchitecture(root))
            return false;

        // Load weights
        if (!loadWeights(root))
            return false;

        DBG("Model loaded successfully!");
        DBG("Layers: " << architecture.numLayers);
        DBG("Channels: " << architecture.channels);
        DBG("Receptive Field: " << architecture.receptiveField);
        DBG("Latency: " << architecture.latencyMs << "ms");

        return true;
    }

    const ModelArchitecture& getArchitecture() const { return architecture; }
    const ModelWeights& getWeights() const { return weights; }

private:
    ModelArchitecture architecture;
    ModelWeights weights;

    bool loadArchitecture(juce::DynamicObject* root)
    {
        auto archObj = root->getProperty("architecture").getDynamicObject();
        if (archObj == nullptr)
            return false;

        architecture.channels = archObj->getProperty("channels");
        architecture.numLayers = archObj->getProperty("num_layers");
        architecture.kernelSize = archObj->getProperty("kernel_size");
        architecture.dilationBase = archObj->getProperty("dilation_base");

        // Load dilations array
        auto dilationsArray = archObj->getProperty("dilations").getArray();
        if (dilationsArray != nullptr)
        {
            for (const auto& val : *dilationsArray)
                architecture.dilations.push_back((int)val);
        }

        // Load model info
        auto infoObj = root->getProperty("model_info").getDynamicObject();
        if (infoObj != nullptr)
        {
            architecture.receptiveField = infoObj->getProperty("receptive_field");
            architecture.latencyMs = infoObj->getProperty("latency_ms");
            architecture.sampleRate = infoObj->getProperty("sample_rate");
        }

        return true;
    }

    bool loadWeights(juce::DynamicObject* root)
    {
        auto weightsObj = root->getProperty("weights").getDynamicObject();
        if (weightsObj == nullptr)
            return false;

        // Load input layer
        auto inputObj = weightsObj->getProperty("input_conv").getDynamicObject();
        if (inputObj != nullptr)
        {
            loadFloatArray(inputObj->getProperty("weight"), weights.inputWeight);
            loadFloatArray(inputObj->getProperty("bias"), weights.inputBias);
        }

        // Load residual blocks
        auto blocksArray = weightsObj->getProperty("residual_blocks").getArray();
        if (blocksArray != nullptr)
        {
            for (const auto& blockVar : *blocksArray)
            {
                auto blockObj = blockVar.getDynamicObject();
                if (blockObj == nullptr)
                    continue;

                LayerWeights layer;
                layer.dilation = blockObj->getProperty("dilation");

                // Load conv weights (3D array)
                auto convWeightArray = blockObj->getProperty("conv_weight").getArray();
                if (convWeightArray != nullptr)
                {
                    load3DFloatArray(convWeightArray, layer.convWeight);
                }

                // Load conv bias
                loadFloatArray(blockObj->getProperty("conv_bias"), layer.convBias);

                weights.residualBlocks.push_back(layer);
            }
        }

        // Load output layer
        auto outputObj = weightsObj->getProperty("output_conv").getDynamicObject();
        if (outputObj != nullptr)
        {
            loadFloatArray(outputObj->getProperty("weight"), weights.outputWeight);
            loadFloatArray(outputObj->getProperty("bias"), weights.outputBias);
        }

        return true;
    }

    void loadFloatArray(const juce::var& arrayVar, std::vector<float>& output)
    {
        auto array = arrayVar.getArray();
        if (array != nullptr)
        {
            output.clear();
            for (const auto& val : *array)
                output.push_back((float)val);
        }
    }

    void load3DFloatArray(juce::Array<juce::var>* array3D,
                          std::vector<std::vector<std::vector<float>>>& output)
    {
        output.clear();
        for (const auto& dim1 : *array3D)
        {
            auto array2D = dim1.getArray();
            if (array2D == nullptr)
                continue;

            std::vector<std::vector<float>> vec2D;
            for (const auto& dim2 : *array2D)
            {
                auto array1D = dim2.getArray();
                if (array1D == nullptr)
                    continue;

                std::vector<float> vec1D;
                for (const auto& val : *array1D)
                    vec1D.push_back((float)val);

                vec2D.push_back(vec1D);
            }
            output.push_back(vec2D);
        }
    }
};

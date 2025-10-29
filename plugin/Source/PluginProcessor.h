/*
  ==============================================================================

    PluginProcessor.h
    Neural Vox Modeler - Real-time guitar amp modeling plugin

  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "ModelLoader.h"
#include <vector>
#include <memory>

//==============================================================================
/**
 * WaveNet-style neural network for real-time audio processing
 */
class NeuralModel
{
public:
    NeuralModel() = default;

    void prepare(const ModelArchitecture& arch, const ModelWeights& weights)
    {
        architecture = arch;
        this->weights = weights;

        // Initialize buffer for receptive field
        contextBuffer.resize(architecture.receptiveField, 0.0f);
        tempBuffer.resize(architecture.channels);

        isReady = true;
    }

    float processSample(float input)
    {
        if (!isReady)
            return input;

        // Shift context buffer (maintain receptive field)
        for (int i = architecture.receptiveField - 1; i > 0; --i)
            contextBuffer[i] = contextBuffer[i - 1];

        contextBuffer[0] = input;

        // Input layer: 1 -> channels
        for (int c = 0; c < architecture.channels; ++c)
        {
            tempBuffer[c] = input * weights.inputWeight[c] + weights.inputBias[c];
        }

        // Residual blocks
        for (int layer = 0; layer < architecture.numLayers; ++layer)
        {
            processResidualBlock(layer);
        }

        // Output layer: channels -> 1
        float output = weights.outputBias[0];
        for (int c = 0; c < architecture.channels; ++c)
        {
            output += tempBuffer[c] * weights.outputWeight[c];
        }

        return output;
    }

    void reset()
    {
        std::fill(contextBuffer.begin(), contextBuffer.end(), 0.0f);
        std::fill(tempBuffer.begin(), tempBuffer.end(), 0.0f);
    }

    int getLatencySamples() const { return isReady ? architecture.receptiveField : 0; }

private:
    ModelArchitecture architecture;
    ModelWeights weights;
    std::vector<float> contextBuffer; // Maintains receptive field for causal processing
    std::vector<float> tempBuffer;    // Channel activations
    bool isReady = false;

    void processResidualBlock(int layerIndex)
    {
        const auto& layer = weights.residualBlocks[layerIndex];
        const int dilation = layer.dilation;
        const int kernelSize = architecture.kernelSize;

        std::vector<float> residual = tempBuffer; // Save for residual connection
        std::vector<float> output(architecture.channels, 0.0f);

        // Dilated causal convolution
        for (int outCh = 0; outCh < architecture.channels; ++outCh)
        {
            float sum = layer.convBias[outCh];

            for (int inCh = 0; inCh < architecture.channels; ++inCh)
            {
                for (int k = 0; k < kernelSize; ++k)
                {
                    // Access past samples with dilation
                    int bufferIdx = k * dilation;
                    if (bufferIdx < contextBuffer.size())
                    {
                        // Use context buffer for causal access
                        sum += layer.convWeight[outCh][inCh][k] * tempBuffer[inCh];
                    }
                }
            }

            // Tanh activation
            output[outCh] = std::tanh(sum);
        }

        // Residual connection
        for (int c = 0; c < architecture.channels; ++c)
        {
            tempBuffer[c] = output[c] + residual[c];
        }
    }
};

//==============================================================================
/**
 * Main plugin processor
 */
class NeuralVoxModelerAudioProcessor : public juce::AudioProcessor
{
public:
    //==============================================================================
    NeuralVoxModelerAudioProcessor();
    ~NeuralVoxModelerAudioProcessor() override;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    //==============================================================================
    // Parameters
    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts; }

    // Model management
    bool loadModelFromFile(const juce::File& file);
    void unloadModel();
    bool isModelLoaded() const { return modelLoaded; }
    juce::String getLoadedModelName() const { return currentModelName; }
    juce::String getLoadedModelInfo() const;
    juce::String getErrorMessage() const { return errorMessage; }
    float getLatencyMs() const { return latencyMs; }

private:
    //==============================================================================
    juce::AudioProcessorValueTreeState apvts;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    NeuralModel neuralModel;
    bool modelLoaded = false;
    float latencyMs = 0.0f;

    // Model state
    juce::File currentModelFile;
    juce::String currentModelName;
    juce::String errorMessage;
    ModelArchitecture currentArchitecture;

    // Parameters
    std::atomic<float>* inputGainParam = nullptr;
    std::atomic<float>* outputGainParam = nullptr;
    std::atomic<float>* mixParam = nullptr;
    std::atomic<float>* bypassParam = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralVoxModelerAudioProcessor)
};

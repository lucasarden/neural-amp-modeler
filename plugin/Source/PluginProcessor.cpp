/*
  ==============================================================================

    PluginProcessor.cpp
    Neural Vox Modeler - Implementation with Dynamic Model Loading

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NeuralVoxModelerAudioProcessor::NeuralVoxModelerAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::mono(), true)
                         .withOutput("Output", juce::AudioChannelSet::mono(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    // Get parameter pointers
    inputGainParam = apvts.getRawParameterValue("inputGain");
    outputGainParam = apvts.getRawParameterValue("outputGain");
    mixParam = apvts.getRawParameterValue("mix");
    bypassParam = apvts.getRawParameterValue("bypass");

    // Model will be loaded dynamically - no hardcoded loading!
}

NeuralVoxModelerAudioProcessor::~NeuralVoxModelerAudioProcessor()
{
}

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
NeuralVoxModelerAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Input Gain: -36dB to +12dB (wide range to match any input level)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "inputGain", "Input Gain", juce::NormalisableRange<float>(-36.0f, 12.0f, 0.1f),
        -18.0f, juce::String(),  // Default to -18dB (good starting point)
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; }));

    // Output Gain: -12dB to +12dB
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "outputGain", "Output Gain", juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f),
        0.0f, juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; }));

    // Mix: 0% to 100%
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "mix", "Mix", juce::NormalisableRange<float>(0.0f, 100.0f, 1.0f), 100.0f,
        juce::String(), juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String((int)value) + " %"; }));

    // Bypass
    params.push_back(
        std::make_unique<juce::AudioParameterBool>("bypass", "Bypass", false));

    return {params.begin(), params.end()};
}

//==============================================================================
bool NeuralVoxModelerAudioProcessor::loadModelFromFile(const juce::File& file)
{
    errorMessage = "";

    // Validate file exists
    if (!file.existsAsFile())
    {
        errorMessage = "File does not exist";
        DBG("Model load failed: File does not exist - " << file.getFullPathName());
        return false;
    }

    // Validate file extension
    if (file.getFileExtension() != ".json")
    {
        errorMessage = "Invalid file type (expected .json)";
        DBG("Model load failed: Invalid file extension - " << file.getFileExtension());
        return false;
    }

    // Load and validate model
    ModelLoader loader;
    if (!loader.loadFromFile(file))
    {
        errorMessage = "Failed to parse JSON file";
        DBG("Model load failed: Failed to parse JSON");
        return false;
    }

    const auto& arch = loader.getArchitecture();

    // Validate model is causal
    auto jsonText = file.loadFileAsString();
    auto json = juce::JSON::parse(jsonText);
    if (json.isObject())
    {
        auto modelInfo = json.getProperty("model_info", juce::var()).getDynamicObject();
        if (modelInfo != nullptr)
        {
            bool isCausal = modelInfo->getProperty("causal");
            if (!isCausal)
            {
                errorMessage = "Model is not causal (cannot use for real-time)";
                DBG("Model load failed: Model is not causal");
                return false;
            }
        }
    }

    // Warn if sample rate mismatch (but allow it)
    if (getSampleRate() > 0 && arch.sampleRate > 0)
    {
        if (std::abs(getSampleRate() - arch.sampleRate) > 1.0)
        {
            DBG("WARNING: Model trained at " << arch.sampleRate << "Hz, but DAW is running at " << getSampleRate() << "Hz");
        }
    }

    // Initialize neural model
    neuralModel.prepare(arch, loader.getWeights());

    // Update state
    currentModelFile = file;
    currentModelName = file.getFileNameWithoutExtension();
    currentArchitecture = arch;
    latencyMs = arch.latencyMs;
    modelLoaded = true;

    // Set plugin latency compensation
    setLatencySamples(neuralModel.getLatencySamples());

    DBG("Model loaded successfully: " << currentModelName);
    DBG("  Layers: " << arch.numLayers);
    DBG("  Channels: " << arch.channels);
    DBG("  Latency: " << latencyMs << "ms");

    return true;
}

void NeuralVoxModelerAudioProcessor::unloadModel()
{
    modelLoaded = false;
    currentModelFile = juce::File();
    currentModelName = "";
    latencyMs = 0.0f;
    setLatencySamples(0);
    neuralModel.reset();

    DBG("Model unloaded");
}

juce::String NeuralVoxModelerAudioProcessor::getLoadedModelInfo() const
{
    if (!modelLoaded)
        return "No model loaded";

    juce::String info;
    info << currentArchitecture.numLayers << " layers, ";
    info << currentArchitecture.channels << " channels, ";
    info << latencyMs << "ms latency";

    return info;
}

//==============================================================================
const juce::String NeuralVoxModelerAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool NeuralVoxModelerAudioProcessor::acceptsMidi() const
{
    return false;
}

bool NeuralVoxModelerAudioProcessor::producesMidi() const
{
    return false;
}

bool NeuralVoxModelerAudioProcessor::isMidiEffect() const
{
    return false;
}

double NeuralVoxModelerAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int NeuralVoxModelerAudioProcessor::getNumPrograms()
{
    return 1;
}

int NeuralVoxModelerAudioProcessor::getCurrentProgram()
{
    return 0;
}

void NeuralVoxModelerAudioProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String NeuralVoxModelerAudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void NeuralVoxModelerAudioProcessor::changeProgramName(int index,
                                                        const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

//==============================================================================
void NeuralVoxModelerAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate);

    // Pre-allocate dry buffer (avoid allocation in processBlock)
    dryBuffer.resize(samplesPerBlock);

    // Reset neural model state
    if (modelLoaded)
        neuralModel.reset();
}

void NeuralVoxModelerAudioProcessor::releaseResources()
{
}

bool NeuralVoxModelerAudioProcessor::isBusesLayoutSupported(
    const BusesLayout& layouts) const
{
    // Only mono in/out
    return layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono() &&
           layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono();
}

void NeuralVoxModelerAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                                   juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);

    juce::ScopedNoDenormals noDenormals();

    // Get parameters
    const float inputGainDb = inputGainParam->load();
    const float outputGainDb = outputGainParam->load();
    const float mixPercent = mixParam->load();
    const bool bypass = bypassParam->load() > 0.5f;

    const float inputGain = juce::Decibels::decibelsToGain(inputGainDb);
    const float outputGain = juce::Decibels::decibelsToGain(outputGainDb);
    const float mix = mixPercent / 100.0f;

    auto* channelData = buffer.getWritePointer(0);
    const int numSamples = buffer.getNumSamples();

    // If bypass OR no model loaded, pass through
    if (bypass || !modelLoaded)
    {
        return;
    }

    // OPTIMIZED: Buffer-based processing instead of sample-by-sample
    // This is the critical performance improvement!

    // Ensure dry buffer is large enough
    if (dryBuffer.size() < static_cast<size_t>(numSamples))
        dryBuffer.resize(numSamples);

    // Save dry signal for wet/dry mixing (no allocation!)
    std::memcpy(dryBuffer.data(), channelData, numSamples * sizeof(float));

    // Apply input gain (vectorized operation)
    juce::FloatVectorOperations::multiply(channelData, inputGain, numSamples);

    // Process through neural network (entire buffer at once!)
    neuralModel.processBuffer(channelData, numSamples);

    // Apply output gain (vectorized operation)
    juce::FloatVectorOperations::multiply(channelData, outputGain, numSamples);

    // Mix wet/dry (vectorized operations)
    if (mix < 1.0f)
    {
        // wet = wet * mix
        juce::FloatVectorOperations::multiply(channelData, mix, numSamples);

        // dry = dry * (1 - mix)
        juce::FloatVectorOperations::multiply(dryBuffer.data(), 1.0f - mix, numSamples);

        // output = wet + dry
        juce::FloatVectorOperations::add(channelData, dryBuffer.data(), numSamples);
    }
}

//==============================================================================
bool NeuralVoxModelerAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* NeuralVoxModelerAudioProcessor::createEditor()
{
    return new NeuralVoxModelerAudioProcessorEditor(*this);
}

//==============================================================================
void NeuralVoxModelerAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    // Save parameters
    auto state = apvts.copyState();

    // Add model file path to state
    if (currentModelFile.existsAsFile())
    {
        state.setProperty("modelPath", currentModelFile.getFullPathName(), nullptr);
    }

    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void NeuralVoxModelerAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

    if (xmlState.get() != nullptr)
    {
        if (xmlState->hasTagName(apvts.state.getType()))
        {
            auto state = juce::ValueTree::fromXml(*xmlState);
            apvts.replaceState(state);

            // Restore model from saved path
            auto modelPath = state.getProperty("modelPath", juce::var()).toString();
            if (modelPath.isNotEmpty())
            {
                juce::File modelFile(modelPath);
                if (modelFile.existsAsFile())
                {
                    loadModelFromFile(modelFile);
                }
                else
                {
                    DBG("Saved model file not found: " << modelPath);
                }
            }
        }
    }
}

//==============================================================================
// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralVoxModelerAudioProcessor();
}

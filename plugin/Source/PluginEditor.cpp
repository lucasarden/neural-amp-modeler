/*
  ==============================================================================

    PluginEditor.cpp
    Neural Vox Modeler - GUI Implementation with Drag & Drop

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NeuralVoxModelerAudioProcessorEditor::NeuralVoxModelerAudioProcessorEditor(
    NeuralVoxModelerAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p),
      inputGainSlider("INPUT GAIN", p.getAPVTS(), "inputGain"),
      outputGainSlider("OUTPUT GAIN", p.getAPVTS(), "outputGain"),
      mixSlider("MIX", p.getAPVTS(), "mix"),
      bypassAttachment(p.getAPVTS(), "bypass", bypassButton)
{
    // Set size
    setSize(550, 450);

    // Title
    addAndMakeVisible(titleLabel);
    titleLabel.setText("NEURAL VOX MODELER", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    titleLabel.setFont(juce::Font(28.0f, juce::Font::bold));

    // Load Model Button
    addAndMakeVisible(loadModelButton);
    loadModelButton.setButtonText("LOAD MODEL");
    loadModelButton.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff4da6ff));
    loadModelButton.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    loadModelButton.onClick = [this] { loadModelButtonClicked(); };

    // Model name display
    addAndMakeVisible(modelNameLabel);
    modelNameLabel.setJustificationType(juce::Justification::centred);
    modelNameLabel.setFont(juce::Font(16.0f, juce::Font::bold));

    // Model info display
    addAndMakeVisible(modelInfoLabel);
    modelInfoLabel.setJustificationType(juce::Justification::centred);
    modelInfoLabel.setFont(juce::Font(12.0f));
    modelInfoLabel.setColour(juce::Label::textColourId, juce::Colour(0xff999999));

    // Status label
    addAndMakeVisible(statusLabel);
    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setColour(juce::Label::textColourId, juce::Colour(0xff999999));
    statusLabel.setFont(juce::Font(11.0f));

    // Sliders
    addAndMakeVisible(inputGainSlider);
    addAndMakeVisible(outputGainSlider);
    addAndMakeVisible(mixSlider);

    // Bypass button
    addAndMakeVisible(bypassButton);
    bypassButton.setButtonText("BYPASS");
    bypassButton.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff1a1a2e));
    bypassButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xffff4d4d));
    bypassButton.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    bypassButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    bypassButton.setClickingTogglesState(true);

    // Initial display update
    updateModelDisplay();

    // Start timer for status updates
    startTimerHz(10);
}

NeuralVoxModelerAudioProcessorEditor::~NeuralVoxModelerAudioProcessorEditor()
{
    stopTimer();
}

//==============================================================================
void NeuralVoxModelerAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Background gradient
    g.fillAll(juce::Colour(0xff0f0f1e));

    auto bounds = getLocalBounds();

    // Header background
    auto headerBounds = bounds.removeFromTop(70);
    g.setGradientFill(juce::ColourGradient(
        juce::Colour(0xff1a1a2e), headerBounds.getX(), headerBounds.getY(),
        juce::Colour(0xff0f0f1e), headerBounds.getX(), headerBounds.getBottom(), false));
    g.fillRect(headerBounds);

    // Border
    g.setColour(juce::Colour(0xff4da6ff).withAlpha(0.3f));
    g.drawRect(getLocalBounds(), 2);

    // Drag & drop overlay
    if (isDraggingFile)
    {
        g.setColour(juce::Colour(0xff4da6ff).withAlpha(0.2f));
        g.fillAll();

        g.setColour(juce::Colour(0xff4da6ff));
        g.setFont(juce::Font(20.0f, juce::Font::bold));
        g.drawText("Drop .json model file here", getLocalBounds(),
                   juce::Justification::centred);
    }
}

void NeuralVoxModelerAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();

    // Title
    titleLabel.setBounds(bounds.removeFromTop(70));

    bounds.removeFromTop(10); // Spacing

    // Load Model Button
    auto buttonBounds = bounds.removeFromTop(40).reduced(120, 0);
    loadModelButton.setBounds(buttonBounds);

    bounds.removeFromTop(10); // Spacing

    // Model name
    modelNameLabel.setBounds(bounds.removeFromTop(25));

    // Model info
    modelInfoLabel.setBounds(bounds.removeFromTop(20));

    bounds.removeFromTop(15); // Spacing

    // Sliders row
    auto sliderBounds = bounds.removeFromTop(140);
    const int sliderWidth = sliderBounds.getWidth() / 3;

    inputGainSlider.setBounds(sliderBounds.removeFromLeft(sliderWidth).reduced(10));
    outputGainSlider.setBounds(sliderBounds.removeFromLeft(sliderWidth).reduced(10));
    mixSlider.setBounds(sliderBounds.removeFromLeft(sliderWidth).reduced(10));

    bounds.removeFromTop(20); // Spacing

    // Bypass button
    auto bypassBounds = bounds.removeFromTop(50).reduced(150, 5);
    bypassButton.setBounds(bypassBounds);

    bounds.removeFromTop(10); // Spacing

    // Status label
    statusLabel.setBounds(bounds.removeFromTop(30));
}

//==============================================================================
// Drag & Drop Implementation
bool NeuralVoxModelerAudioProcessorEditor::isInterestedInFileDrag(
    const juce::StringArray& files)
{
    // Only accept .json files
    for (const auto& file : files)
    {
        if (juce::File(file).hasFileExtension(".json"))
            return true;
    }
    return false;
}

void NeuralVoxModelerAudioProcessorEditor::filesDropped(const juce::StringArray& files,
                                                         int x, int y)
{
    juce::ignoreUnused(x, y);
    isDraggingFile = false;
    repaint();

    // Load first .json file found
    for (const auto& filePath : files)
    {
        juce::File file(filePath);
        if (file.hasFileExtension(".json"))
        {
            bool success = audioProcessor.loadModelFromFile(file);
            updateModelDisplay();

            if (!success)
            {
                juce::AlertWindow::showMessageBoxAsync(
                    juce::AlertWindow::WarningIcon, "Error Loading Model",
                    "Failed to load model:\n" + audioProcessor.getErrorMessage(),
                    "OK");
            }

            break; // Only load first file
        }
    }
}

void NeuralVoxModelerAudioProcessorEditor::fileDragEnter(
    const juce::StringArray& files, int x, int y)
{
    juce::ignoreUnused(files, x, y);
    isDraggingFile = true;
    repaint();
}

void NeuralVoxModelerAudioProcessorEditor::fileDragExit(const juce::StringArray& files)
{
    juce::ignoreUnused(files);
    isDraggingFile = false;
    repaint();
}

//==============================================================================
void NeuralVoxModelerAudioProcessorEditor::loadModelButtonClicked()
{
    // Create file chooser
    fileChooser = std::make_unique<juce::FileChooser>(
        "Select Model File", juce::File::getSpecialLocation(
                                 juce::File::userDocumentsDirectory),
        "*.json");

    auto flags = juce::FileBrowserComponent::openMode |
                 juce::FileBrowserComponent::canSelectFiles;

    fileChooser->launchAsync(
        flags, [this](const juce::FileChooser& chooser)
        {
            auto file = chooser.getResult();

            if (file.existsAsFile())
            {
                bool success = audioProcessor.loadModelFromFile(file);
                updateModelDisplay();

                if (!success)
                {
                    juce::AlertWindow::showMessageBoxAsync(
                        juce::AlertWindow::WarningIcon, "Error Loading Model",
                        "Failed to load model:\n" + audioProcessor.getErrorMessage(),
                        "OK");
                }
            }
        });
}

void NeuralVoxModelerAudioProcessorEditor::updateModelDisplay()
{
    if (audioProcessor.isModelLoaded())
    {
        // Model loaded - show green status
        modelNameLabel.setText(audioProcessor.getLoadedModelName(),
                               juce::dontSendNotification);
        modelNameLabel.setColour(juce::Label::textColourId,
                                 juce::Colour(0xff4dff4d));

        modelInfoLabel.setText(audioProcessor.getLoadedModelInfo(),
                               juce::dontSendNotification);

        statusLabel.setText("Drag & drop .json file to switch models",
                            juce::dontSendNotification);
    }
    else
    {
        // No model loaded - show warning
        modelNameLabel.setText("No Model Loaded", juce::dontSendNotification);
        modelNameLabel.setColour(juce::Label::textColourId,
                                 juce::Colour(0xffff9933));

        modelInfoLabel.setText("Click 'Load Model' or drag & drop a .json file",
                               juce::dontSendNotification);

        statusLabel.setText("Audio will pass through (bypass) until model is loaded",
                            juce::dontSendNotification);
    }
}

void NeuralVoxModelerAudioProcessorEditor::timerCallback()
{
    // Refresh display if model state changed
    static bool lastModelState = false;
    bool currentModelState = audioProcessor.isModelLoaded();

    if (lastModelState != currentModelState)
    {
        updateModelDisplay();
        lastModelState = currentModelState;
    }
}

/*
  ==============================================================================

    PluginEditor.h
    Neural Vox Modeler - GUI with Dynamic Model Loading

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"

//==============================================================================
/**
 * Custom rotary slider with label
 */
class RotarySliderWithLabel : public juce::Component
{
public:
    RotarySliderWithLabel(const juce::String& labelText,
                          juce::AudioProcessorValueTreeState& apvts,
                          const juce::String& paramID)
        : attachment(apvts, paramID, slider)
    {
        slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 20);
        slider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::white);
        slider.setColour(juce::Slider::textBoxBackgroundColourId,
                         juce::Colours::transparentBlack);
        slider.setColour(juce::Slider::thumbColourId, juce::Colour(0xff4da6ff));
        slider.setColour(juce::Slider::rotarySliderFillColourId,
                         juce::Colour(0xff4da6ff));
        slider.setColour(juce::Slider::rotarySliderOutlineColourId,
                         juce::Colour(0xff1a1a2e));

        addAndMakeVisible(slider);
        addAndMakeVisible(labelComponent);
        labelComponent.setText(labelText, juce::dontSendNotification);
        labelComponent.setJustificationType(juce::Justification::centred);
        labelComponent.setColour(juce::Label::textColourId, juce::Colours::white);
    }

    void resized() override
    {
        auto bounds = getLocalBounds();
        labelComponent.setBounds(bounds.removeFromTop(20));
        slider.setBounds(bounds);
    }

private:
    juce::Slider slider;
    juce::Label labelComponent;
    juce::AudioProcessorValueTreeState::SliderAttachment attachment;
};

//==============================================================================
/**
 * Plugin editor GUI with drag & drop model loading
 */
class NeuralVoxModelerAudioProcessorEditor : public juce::AudioProcessorEditor,
                                              public juce::FileDragAndDropTarget,
                                              private juce::Timer
{
public:
    NeuralVoxModelerAudioProcessorEditor(NeuralVoxModelerAudioProcessor&);
    ~NeuralVoxModelerAudioProcessorEditor() override;

    //==============================================================================
    void paint(juce::Graphics&) override;
    void resized() override;

    //==============================================================================
    // Drag & Drop
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;

private:
    void timerCallback() override;
    void loadModelButtonClicked();
    void updateModelDisplay();

    NeuralVoxModelerAudioProcessor& audioProcessor;

    // Title
    juce::Label titleLabel;

    // Model loading
    juce::TextButton loadModelButton;
    juce::Label modelNameLabel;
    juce::Label modelInfoLabel;
    juce::Label statusLabel;

    // Drag & drop visual feedback
    bool isDraggingFile = false;

    // Controls
    RotarySliderWithLabel inputGainSlider;
    RotarySliderWithLabel outputGainSlider;
    RotarySliderWithLabel mixSlider;
    juce::TextButton bypassButton;

    // Bypass attachment
    juce::AudioProcessorValueTreeState::ButtonAttachment bypassAttachment;

    // File chooser
    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralVoxModelerAudioProcessorEditor)
};

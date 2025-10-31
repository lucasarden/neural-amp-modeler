cmake_minimum_required(VERSION 3.15)

project(NeuralVoxModeler VERSION 1.0.0)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        7.0.9
)
FetchContent_MakeAvailable(JUCE)

# Plugin formats to build
set(FORMATS VST3 Standalone)

# If you want AU (macOS) or AAX, add them here:
# set(FORMATS VST3 AU Standalone)

# Add plugin target
juce_add_plugin(NeuralVoxModeler
    COMPANY_NAME "YourCompany"
    PLUGIN_MANUFACTURER_CODE Yoco
    PLUGIN_CODE Nvm1
    FORMATS ${FORMATS}
    PRODUCT_NAME "Neural Vox Modeler"
    PLUGIN_NAME "Neural Vox Modeler"
    DESCRIPTION "Real-time neural guitar amp modeling"

    # Plugin characteristics
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE

    # Copy plugin after build
    COPY_PLUGIN_AFTER_BUILD TRUE

    # VST3 categories
    VST3_CATEGORIES Fx Distortion
)

# Source files
target_sources(NeuralVoxModeler PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/ModelLoader.h
)

# Compile definitions
target_compile_definitions(NeuralVoxModeler PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
)

# Link JUCE modules
target_link_libraries(NeuralVoxModeler PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_audio_plugin_client
    juce::juce_dsp
PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# OPTIMIZATION FLAGS for maximum performance
if(MSVC)
    # Microsoft Visual C++ optimizations
    target_compile_options(NeuralVoxModeler PRIVATE
        /O2          # Maximum optimization
        /Oi          # Enable intrinsic functions
        /Ot          # Favor fast code
        /arch:AVX2   # Enable AVX2 instructions (SIMD)
        /fp:fast     # Fast floating-point model
        /GL          # Whole program optimization
    )
    target_link_options(NeuralVoxModeler PRIVATE
        /LTCG        # Link-time code generation
    )
else()
    # GCC/Clang optimizations
    target_compile_options(NeuralVoxModeler PRIVATE
        -O3                 # Maximum optimization
        -ffast-math         # Fast floating-point (trade precision for speed)
        -march=native       # Optimize for CPU architecture
        -mtune=native       # Tune for CPU architecture
        -funroll-loops      # Unroll loops
        -fno-signed-zeros   # Assume no signed zeros
        -fno-trapping-math  # Don't trap on math errors
    )
endif()

# Copy resources (model.json) to build directory
add_custom_command(TARGET NeuralVoxModeler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:NeuralVoxModeler>/Resources"
    COMMENT "Creating Resources directory"
)

# Print useful info
message(STATUS "==============================================")
message(STATUS "Neural Vox Modeler Plugin")
message(STATUS "==============================================")
message(STATUS "Building formats: ${FORMATS}")
message(STATUS "CMake Generator: ${CMAKE_GENERATOR}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==============================================")
message(STATUS "After building:")
message(STATUS "1. Export your model weights:")
message(STATUS "   python src/export_plugin_weights.py --model models/realtime_6layer/best_model.pt")
message(STATUS "2. Copy model JSON to:")
message(STATUS "   plugin/build/NeuralVoxModeler_artefacts/*/Resources/model.json")
message(STATUS "3. The VST3 will be in:")
message(STATUS "   plugin/build/NeuralVoxModeler_artefacts/VST3/")
message(STATUS "==============================================")

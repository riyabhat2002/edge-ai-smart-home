# Edge AI Smart Home Gateway
### A privacy-first smart home gateway running local AI on Raspberry Pi

### Using Raspberry Pi 4 to create a privacy first smart home gateway that runs entirely on local hardware with no cloud dependency. It will use an AI model to understand natural language voice commands and control smart home devices

## Functionalitites for MVP:
- A local LLM that understands commands
- MQTT device control
- A dashboard to keep track of the commands received and action taken

## Tech Stack
- Hardware: Raspberry Pi 4
- Inference Engine: llama.cpp
- Model: Llama-3.2-1B-Instruct-Q4_K_M (4-bit quantization)
- Protocol - MQTT for lightweight device orchestration
- OS - Raspberry Pi OS

## Development Log
### 22nd Feb:
- Set up Raspberry Pi 4 with Raspberry Pi OS
- Configured ssh access
- Added Llama.cpp as a submodule
- Downloaded Llama-3.2-1B-Instruct-Q4_K_M model
- model works best with the following flags -> -c 512 --no-kv-offload --no-mmap --threads 3 -ctk q8_0 -ctv q8_0 (0.1 t/s to 4.7 t/s)

### 26th Feb:
- model loading, context init, MQTT setup

### 28th Feb:
- full inference pipeline and generation loop setup
- getting the output in JSON format

### 1st Mar
- Modularized code
- Added MQTTClient by implementing mosquitto_loop_start and condition_variable synchronization to replace manual polling, successfully resolving a race-condition-induced segmentation fault and achieving thread-safe asynchronous messaging for the inference gateway.
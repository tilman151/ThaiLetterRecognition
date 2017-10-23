################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/LetterCandidate.cpp \
../src/SWT.cpp \
../src/ThaiLetterRecognition.cpp \
../src/util.cpp 

OBJS += \
./src/LetterCandidate.o \
./src/SWT.o \
./src/ThaiLetterRecognition.o \
./src/util.o 

CPP_DEPS += \
./src/LetterCandidate.d \
./src/SWT.d \
./src/ThaiLetterRecognition.d \
./src/util.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -I/usr/local/include/opencv -I/usr/include/boost -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



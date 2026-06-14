# Lesson 001 — Milestone 0: the scaffold

**Date:** 2026-06-14
**Goal:** Get the scaffold compiling, run `--device-info`, and understand every line of `metal_ctx.mm`
and `main.cpp`.

## What I did
  Created the main.cpp and metal context, ran the set up script, ran make

## 1. Why metal_ctx.mm is .mm, not .cpp
metal_ctx_mm is .mm instead of .cpp to allow the a bridge between apple hardware via their gpu api   
AND an agnostic c++ program at the main.cpp layer.

Specific contructs pertaining to objective c++ are   
info.name which is a std::string however [device name] is a NSString* - which needs to be parsed to the UTF8String method -> const char* so c and therefore c++ can        
understand it, the second one is the supportsFamily:MTLGPUFamilyApple7 syntax which is roughly          
equivalent to supportsFamily.MTLGPUFamilyApple7() as an api in most other languages. Both are objective
c++ native.

## 2. What id<MTLDevice> means
id<MTLDevice> is a type - pointer to any object conforming to the MTLDevice contract

## 3. The two memory systems
We have the objective c++ memory management system and the c++ one. ARC (objective c++)              
automatically frees the device and queue parameters when delete ctx destroys the struct however by =    
nil'ing them it frees them a moment earlier - not load-bearing

## 4. MTLCreateSystemDefaultDevice
MTLCreateSystemDefault()        
returns the default gpu - intended for multi-gpu apple era (intel), on silicon will always return       
AppleFamily - e.g Apple7, the hardware decides this default.

## 5. main.cpp control flow
main.cpp's flow is to first verify the  
arg count, if less than 2. prints the first index of the arg vector and returns failure, if not it      
compares the 2nd index of the arg vector to --version, prints rigel, returns exit. if not it compares   
the argv's 2nd index to the --device-info flag, create the metal context, if it is null pointer,       
prints error, returns failure. if not returns metal context info and then destroys it to manage memory,
then returns success.                                                                                   


## Next
  <!-- Compartment 2: tensors & memory (RAII, ownership, rule of five, MTLBuffer) -->
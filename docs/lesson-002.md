# Lesson 002 — Compartment 2: tensors & ownership

**Date:** 2026-06-14

## What a tensor is
  <!-- the three essentials: shape, data, dtype — and which one is the "resource" -->
A tensor is a flattened matrix used for inference, to get it from matrix form to a tensor we need 3 bits of information from the matrix
the shape {n x m}, the data - the numbers, and the data type - f32, f16, i8, i4. The resource is the data - the raw std::byte*

## Why derive size instead of storing it
Don't store anything you can derive, means the parameters are the single source of truth and nothing will cause variation there

## RAII
  <!-- acquire in constructor, release in destructor; scope -> automatic destruction, LIFO -->
Resource Acquisition is Initialization - the constructor automatically acquires the resource using new[] the moment the object is created.
The destructor automatically releases the resource using delete[] the moment the object dies. Automatic destruction is tied to the
scope an object sits in, and therefore its lifetime. Meaning when execution passes an object, it automatially destroys it.
When you create multiple objects in a function C++ destroys them in reverse order of their creation (LIFO)

## The rule of five
  <!-- why a destructor-only resource type is dangerous (the double-free) -->
  <!-- copy = delete (turns runtime crash into compile error); move ctor vs move assign (the delete[]
  difference you just explained) -->
A destructor-only resource is dangerous because if you only write a destructor and not a copy constructor or copy assigment operator
then C++ automatically writes them for you in a shallow copy way. This means they copy the pointer, so both objects point at
the same buffer. This results in run-time crashes called double-free's

If you want to prevent copying entirely, you can disable copy by deleting the copy operations. 

When enabling move semantics the difference between the constructor and assignment operator comes down to memory ownership.

The move constructor builds a new object so doesn't need to call delete. It steals the pointer from the temp object.
The move assignment operation does need to call delete because it is overwriting an existing object. It already owns an active allocation so
must call on its own memory first to prevent a memory leak before stealing the pointer from the temporary object

## What tripped me up / what I'd forget
  <!-- honest notes: declaration-before-use ordering; method-vs-cast habit; etc. -->
Declaration before use, some Python syntax leaking in


## Open / next session
  <!-- ASan can't run on this Intel laptop (verify "no leaks" on the M1); still TODO: fill + read-back
  methods, then the ASan leak check -->
  Fill and read back methods, verify no leaks on M1
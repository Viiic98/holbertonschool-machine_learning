#!/usr/bin/env python3
""" QA loop """
while True:
    question = input("Q: ")
    bye = ['exit', 'quit', 'goodbye', 'bye']
    if question.lower() in bye:
        print("A: Goodbye")
        break
    print("A: ")

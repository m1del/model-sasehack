from transformers import pipeline

unmasker = pipeline('fill-mask', model='distilroberta-base')

unmasker("I am storing chicken in my refrigerator at 38 degrees fahrenheit, it will spoil in <mask> days.")
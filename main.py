from skipGram import SkipGram, text2sentences

sentences = text2sentences('train.txt')
print(sentences[0])
sg = SkipGram(sentences)
sg.train(2)


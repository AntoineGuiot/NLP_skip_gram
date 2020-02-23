from skipGram import SkipGram, text2sentences

sentences = text2sentences('train_200000.txt')
print(sentences[0])
sg = SkipGram(sentences)
sg.train(2)


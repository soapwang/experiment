
import os
import codecs

address = r'C:\Users\python\Desktop\cc\test.txt'
pos=[]
neg=[]

def main():
    with codecs.open(address,'r','utf-8') as f:
        text = f.readlines()
        for i in text:
            line = i.split(' ')
            if line[0]== '__label__1':
                neg.append(' '.join(line[1:]))
            else:
                pos.append(' '.join(line[1:]))

    with codecs.open('./pos.txt','w','utf-8')as f:
        for i in pos:
            f.write(i)

    with codecs.open('./neg.txt','w','utf-8')as f:
        for i in neg:
            f.write(i)

if __name__ == "__main__":
    main()



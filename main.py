# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:24:36 2013

@author: han
"""
import numpy as np
import pylab as plt
import math
import copy
from scipy import interpolate
from scipy import signal

class xl:    
    def __init__(self, initListFromOutSide, startNO = 0):
        self.date = list()
        self.date.extend(initListFromOutSide)
        self.indexOfStart = startNO
        self.indexOfEnd = (len(self.date) - 1) + startNO
    
    def shitf(self, n):
        self.indexOfStart = -n
        self.indexOfEnd  = self.indexOfEnd - n
         
    def mul(self, n):
        if len(n) == 1:
            for i in xrange(len(self.date)):
                self.date[i] *= n[0]
        else:
            for i in xrange(len(self.date)):
                self.date[i] *= n[i]
            
         
    def __add__(self, other):
        beginN = min(self.indexOfStart, other.indexOfStart)
        endN = max(self.indexOfEnd, other.indexOfEnd)
        ans = list()
        for i in xrange(beginN, (endN+1)):
            tem = 0;
            if i >= self.indexOfStart and i<= self.indexOfEnd:
                tem += self.date[i-self.indexOfStart]
            if i >= other.indexOfStart and i<= other.indexOfEnd:
                tem += other.date[i-other.indexOfStart]
            ans.append(tem)
        return xl(ans, beginN)
        
    def reNum(self):
        for i in xrange(len(self.date)):
            self.date[i] = -self.date[i]
            
def filterLen(sequence, index):
    if index >= 0 and index < len(sequence):
        return sequence[index]
    return 0
    
def changeLen(ListOfInput, Xmin, Xmax):
    return [ListOfInput[i%(len(ListOfInput))] for i in xrange(Xmin, Xmax+1)]
            
def linearConvolution(sequenceA, sequenceB):
    newSequence = list()
    for i in xrange((len(sequenceA) + len(sequenceB) - 1)):
        tem = 0
        for j in xrange(len(sequenceA)):
            tem += filterLen(sequenceA, j)*filterLen(sequenceB, i-j)
        newSequence.append(tem)
    return newSequence

def circularConvolution(sequenceA, sequenceB, N):
    if len(sequenceA) > N or len(sequenceB) > N:
        print "err"
        return
    else:
        sequenceA.extend([0 for i in xrange(N-len(sequenceA))])
        sequenceB.extend([0 for i in xrange(N-len(sequenceB))])
        return [sum([sequenceA[x%len(sequenceA)]*sequenceB[(i-x)%len(sequenceB)] for x in xrange(N)]) for i in xrange(N) ]
        
def uFunction(number):
    if number >= 0:
        return 1
    else:
        return 0
        
def deltaFunction(number):
    if number == 0:
        return 1
    else:
        return 0

def plot(x, y):
    plt.scatter(x, y)
    plt.vlines(x, [0], y)
    plt.ylim((min(y)-abs(min(y)*0.1)),max(y)+max(y)*0.1)
    plt.hlines(0, x[0]-1, x[x.shape[0]-1]+1)
    plt.xlim(x[0]-1,x[x.shape[0]-1]+1)
    plt.show()
    
def plotT(x, y, plt):
    plt.scatter(x, y)
    plt.vlines(x, [0], y)
    plt.ylim((min(y)-abs(min(y)*0.1)),max(y)+max(y)*0.1)
    plt.hlines(0, x[0]-1, x[x.shape[0]-1]+1)
    plt.xlim(x[0]-1,x[x.shape[0]-1]+1)
    plt.grid()

def ex1_1_1():
    y = list()
    n = [i for i in xrange(0,21)]
    for i in n:
        y.append(i*(uFunction(i)-uFunction(i-10))+10*math.exp(-0.3*(i-10))*((uFunction(i-10)-uFunction(i-20))))
    return np.array(n), np.array(y)

def ex1_1_2():
    x = [i for i in xrange(-10,10)]
    y = [i for couter in xrange(4) for i in xrange(5, 0, -1)]
    return np.array(x), np.array(y)
    
def ex1_2_1():
    oldX = [1, -2, 4, 6, -5, 8, 10]
    a = xl(oldX)
    b = xl(oldX)
    c = xl(oldX)
    a.mul([5])
    a.shitf(5)
    b.mul([4])
    b.shitf(4)
    c.mul([3])
    d = a + b + c
    return np.array([i for i in xrange(d.indexOfStart,d.indexOfEnd+1)]), np.array(d.date)
    
def ex1_2_2():
    oldX = changeLen([1, -2, 4, 6, -5, 8, 10], -20, 20)
    a = xl(oldX, -20)
    aPre = [2*np.exp(0.5*i) for i in xrange(-20, 21)]
    a.mul(aPre)
    b = xl(oldX, -20)
    
    bPre = [np.cos(0.1*np.pi*i) for i in xrange(-20, 21)]
    b.shitf(2)
    b.mul(bPre)
    c = a + b
    x,y = np.array([i for i in xrange(c.indexOfStart, c.indexOfEnd+1)]), np.array(c.date)
    plt.scatter(x, y)
    plt.vlines(x, [0], y)
    plt.hlines(0, x[0]-1, x[x.shape[0]-1]+1)
    plt.ylim(-20, 60)
    plt.xlim(-20, 20)
    plt.show()
    

def ex1_3_1():
    y = list()
    for i in xrange(-5, 31):
        y.append(i*(uFunction(i)-uFunction(i-10))+(20-i)*(uFunction(i-10)-uFunction(i-20)))
    a = xl(y)
    b = xl(y)
    b.shitf(-1)
    b.reNum()
    c = a+b
    return np.arange(-5, 32), np.array(c.date)
    
#ex1_2_2()

def sinc_interp(x, s, u):
    
    T = s[1] - s[0]
    
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y
    
def ex2_1(simpleTime):
    one = plt.subplot(311)
    two = plt.subplot(312)
    three = plt.subplot(313)
    t = np.linspace(0, 1, int(1/simpleTime)+1)
    tt = np.linspace(0, 1, 10000)
    thX = np.linspace(0, 1, 1000)
    y = np.cos(np.pi*20*t)
    yy = np.cos(np.pi*20*tt)
    plt.sca(one)
    plt.scatter(t, y)
    plt.plot(tt, yy)
    plt.sca(two)
    twoY =  sinc_interp(y, t, thX)
    plt.plot(thX, twoY)
    
    plt.sca(three)
    
    func = interpolate.interp1d(t, y, kind = 'cubic')
    thY = func(thX)
    plt.plot(thX, thY)
    
    plt.show()
    
    

def ex2_2():
    x1 = [0.8**i for i in xrange(10)]
    a = xl([1])
    b = xl([1])
    b.shitf(-2)
    x2 = a + b
    one = plt.subplot(221)
    two = plt.subplot(222)
    three = plt.subplot(212)
    plt.sca(one)
    plotT(np.arange(10), np.array(x1), plt)
    plt.sca(two)
    plotT(np.arange(3), np.array(x2.date), plt)
    plt.sca(three)
    ans = linearConvolution(x1, x2.date)
    plotT(np.array([i for i in xrange(len(ans))]), np.array(ans), plt)
    plt.show

def ex2_3():
    x1 = [i for i in xrange(1, 6)]
    x2 = [-1, 3, 0, -2, -2, 1]
    lineX =  linearConvolution(copy.deepcopy(x1), copy.deepcopy(x2))
    CirX6 = circularConvolution(copy.deepcopy(x1), copy.deepcopy(x2), 6)
    CirX10 = circularConvolution(copy.deepcopy(x1), copy.deepcopy(x2), 10)
    CirX12 = circularConvolution(copy.deepcopy(x1), copy.deepcopy(x2), 12)
    image1 = subplot(2, 3, 1)
    image2 = subplot(2, 3, 2)
    image3 = subplot(2, 3, 3)
    image4 = subplot(2, 3, 4)
    image5 = subplot(2, 3, 5)
    image6 = subplot(2, 3, 6)
    plt.sca(image1)
    plotT(np.arange(0, 5), np.array(x1), plt)
    plt.sca(image2)
    plotT(np.arange(0, 6), np.array(x2), plt)    
    plt.sca(image3)
    plotT(np.arange(0, len(CirX6)), np.array(CirX6), plt)
    plt.sca(image4)
    plotT(np.arange(0, len(CirX10)), np.array(CirX10), plt)
    plt.sca(image5)
    plotT(np.arange(0, len(CirX12)), np.array(CirX12), plt)
    plt.sca(image6)
    plotT(np.arange(0, len(lineX)), np.array(lineX), plt)
    plt.show()
    
def ex3_1(simpleTime = 0.5):
    y = np.array([1, 3, 5, 1, 1])
    D = 2*np.pi/(len(y)*simpleTime)
    yy = np.fft.fftshift(np.fft.fft(y))
    image1 = subplot(2, 1, 1)
    image2 = subplot(2, 1, 2)
    plt.sca(image1)
    plotT(np.array([i*D for i in xrange(-2, 3)]), np.abs(np.array(yy)), plt)
    plt.sca(image2)
    plotT(np.array([i*D for i in xrange(-2, 3)]), np.angle(np.array(yy)), plt)
    plt.show()
    
def ex3_2(fftSize=512, oututSize = 2048):
    samplingRate = 80000
    t = np.arange(0, 1.0, 1.0/samplingRate)
    x = np.cos(2*np.pi*20000*t) + 2*np.cos(2*np.pi*21000*t)
    xs = np.fft.fft(x[:fftSize], oututSize)
    image1 = subplot(2, 1, 1)
    image2 = subplot(2, 1, 2)
    plt.sca(image1)
    plotT(np.array([i for i in xrange(fftSize)]), np.array(x[:fftSize]), plt)
    plt.sca(image2)
    plotT(np.array([i for i in xrange(oututSize)]), np.abs(np.array(xs)), plt)
    plt.show()


def ex4_1():
    wp = 0.2*np.pi
    ws = 0.3*np.pi
    fs = 4000.0
    T = 1/fs
    Wp = wp/T
    Ws = ws/T
    n, x = signal.buttord(Wp, Ws, 2, 40, analog=True)
    b, a = signal.butter(n, x, analog=True)
    z, p, k = signal.tf2zpk(b, a)
    print z, p, k
    


def ex4_2():
    wp = 0.2*np.pi
    ws = 0.3*np.pi
    fs = 4000.0
    T = 1/fs
    Wp = (2/T)*np.tan(wp/2);
    Ws = (2/T)*np.tan(ws/2);
    n, x = signal.buttord(Wp, Ws, 2, 40, analog=True)
    b, a = signal.butter(n, x, analog=True)
    b2, a2 = signal.bilinear(b, a, 4000)
    w, h = signal.freqz(b2, a2, worN=10000)
    
    image1 = subplot(2, 1, 1)
    image2 = subplot(2, 1, 2)
    
    plt.sca(image1)
    plt.plot(w*fs/2/np.pi, abs(h))
    plt.sca(image2)
    plt.plot(w*fs/2/np.pi, np.angle(h))
    plt.show() 
    
    
        
        
    

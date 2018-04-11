#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:30:22 2017

@author: jabong
"""

import numpy as np

class dtree:
    
    def __init__(self):
        """ Constructor """
    
    def read_data(self, filename):
        fid = open(filename, 'r')
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(','))
        fid.close()
        
        self.featurenames = data[0]
        self.featurenames = self.featurenames[:-1]
        data = data[1:]
        self.classes = []
        for d in range(len(data)):
            self.classes.append(data[d][-1])
            data[d] = data[d][:-1]
        return data, self.classes, self.featurenames
    
    def classify(self, tree, datapoint):
        if type(tree) == type("string"):
            return tree
        else:
            a = tree.keys()[0]
            for i in range(len(self.featurenames)):
                if self.featurenames[i] == a:
                    break;
            try:
                t = tree[a][datapoint[i]]
                return self.classify(t, datapoint)
            except:
                return None
    
    def classifyAll(self, tree, data):
        results = []
        for i in range(len(data)):
            results.append(self.classify(tree, data[i]))
        return results
    
    def make_tree(self, data, classes, featureNames, maxlevel=-1, level=0, forest=0):
        nData = len(data)
        nFeatures = len(data[0])
        
        try:
            self.featurenames
        except:
            self.featurenames = featureNames
        
        newClasses = []
        for aclass in classes:
            if newClasses.count(aclass) == 0:
                newClasses.append(aclass)
        
        frequency = np.zeros(len(newClasses))
        
        totalEntropy = 0
        totalGini = 0
        index = 0
        for aclass in newClasses:
            frequency[index] = classes.count(aclass)
            totalEntropy += self.calc_entropy(float(frequency[index])/nData)
            totalGini += (float(frequency[index])/nData)**2
            index += 1
        
        totalGini = 1 - totalGini
        default = classes[np.argmax(frequency)]
        
        if nData == 0 or nFeatures == 0 or (maxlevel>=0 and level>maxlevel):
            return default
        elif classes.count(classes[0]) == nData:
            return classes[0]
        else:
            gain = np.zeros(nFeatures)
            ggain = np.zeros(nFeatures)
            featureSet = range(nFeatures)
            if forest != 0:
                np.random.shuffle(featureSet)
                featureSet = featureSet[0:forest]
            for feature in featureSet:
                g, gg = self.calc_info_gain(data, classes, feature)
                gain[feature] = totalEntropy - g
                ggain[feature] = totalGini - gg
            
            bestFeature = np.argmax(gain)
            tree = {featureNames[bestFeature]:{}}
            
            values = []
            for datapoint in data:
                if datapoint[feature] not in values:
                    values.append(datapoint[bestFeature])
            
            for value in values:
                newData = []
                newClasses = []
                index = 0
                for datapoint in data:
                    if datapoint[bestFeature] == value:
                        if bestFeature==0:
                            newdatapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature==nFeatures:
                            newdatapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            newdatapoint = datapoint[:bestFeature]
                            newdatapoint.extend(datapoint[bestFeature+1:])
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature+1:])
                        newData.append(newdatapoint)
                        newClasses.append(classes[index])
                    index += 1
                
                subtree = self.make_tree(newData, newClasses, newNames, maxlevel, level+1, forest)
                
                tree[featureNames[bestFeature]][value] = subtree
            return tree
    
    def printTree(self, tree, name):
        if type(tree) == dict:
            print name, tree.keys()[0]
            for item in tree.values()[0].keys():
                print name, item
                self.printTree(tree.values()[0][item], name + "\t")
        else:
            print name, "\t->\t", tree
    
    def calc_entropy(self, p):
        if p!=0:
            return -p * np.log2(p)
        else:
            return 0
    
    def calc_info_gain(self, data, classes, feature):
        gain = 0
        ggain = 0
        nData = len(data)
        
        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])
        
        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        gini = np.zeros(len(values))
        valueIndex = 0
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature]==value:
                    featureCounts[valueIndex]+=1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1
            
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass)==0:
                    classValues.append(aclass)
            
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex]+=1
                classIndex += 1
            
            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex])/np.sum(classCounts))
                gini[valueIndex] += (float(classCounts[classIndex])/np.sum(classCounts))**2
            
            gain = gain + float(featureCounts[valueIndex])/nData * entropy[valueIndex]
            ggain = ggain + float(featureCounts[valueIndex])/nData * gini[valueIndex]
            valueIndex += 1
        return gain, 1-ggain

            
    
    
        
        
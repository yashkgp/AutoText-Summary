#!/usr/bin/python

import glob
import io

class Parse():

    def dataFromFile(self, path):
        '''
        Read textual data from a file
        Arguments:
            path: path of the file (syntax is os dependent)
        Returns:
            data: a string containing data of the file
        '''
        text = ""
        input_files = glob.glob(path)
        for file in input_files:
            with open(file, 'r') as f:
                text = f.read()
        return text

    def writeDataToFile(self, pathToFile, data):
        File = io.open(pathToFile, 'w')
        File.write(data)
        File.close()

    def writeListDataToFile(self, pathToFile, Listdata):
        File = io.open(pathToFile, 'a')
        for item in Listdata:
            File.write(item)
        File.close()

    def dataFromFolder(self, path):
        '''
        Read textual data from every file within a Folder
        Arguments:
            path: path of the folder containing the files (syntax is os dependent)
        Returns:
            data: a list containing data of each file
        '''
        text = ""
        data = []
        counter = -1
        input_files = glob.glob(path)
        for file in input_files:
            counter += 1
            with open(file, 'r') as f:
                text = f.read()
            data.append(text)
        return data

# metadata_parser.py
# Mike Zheng and Heidi He
# 5/13/19
#
# read directory of xml files and parse out the metadata
# creator, type, material, year
#
# python3 metadata_parser.py /Users/xiaoyuezheng/Desktop/Rijksmuseum_data_raw/metadata_subset
#
# sort using shell command: sort -t"," -k1n,1 ../data/metadata_first.csv > ../data/metadata_first_reorder.csv
# sort -t"," -k1n,1 ../data/metadata_subset_first.csv > ../data/metadata_subset_first_reorder.csv
import sys
import os
import xml.etree.ElementTree as ET
import numpy as np

# return a list of file paths of a given directory
def readdir(dir):
    filelist = []
    for root, directories, filenames in os.walk(dir):
        # for directory in directories:
        #     print(os.path.join(root, directory))
        for filename in filenames:
            if filename[0] != '.':
                filelist.append(os.path.join(root,filename))
    return filelist

# parse xml file
# return a list of list [[creator],[type],[material],[year]]
def parseXML(xmlfile):

    namespace_format = '<ns1 xmlns:oai_dc="a"><ns2 xmlns:dc="b">{}</ns2></ns1>'

    # create element tree object
    fp_xml = open(xmlfile,"r")
    s = fp_xml.read()
    s_formatted = namespace_format.format(s)
    fp_xml.close()

    root = ET.fromstring(s_formatted)

    items = [[],[],[],[]]

    for item in root.findall('./ns2/record/metadata/{a}dc/'):
        tag = item.tag[3:]
        if tag == "creator":
            if item.text is not None:
                items[0].append("".join(item.text.split(":")[1].strip().split(",")))
            else:
                items[0].append("")
        elif tag == "type":
            if item.text is not None:
                items[1].append(item.text.strip())
            else:
                items[1].append("")
        elif tag == "format":
            if item.text is not None:
                words = item.text.split(":")
                if words[0]=="materiaal":
                    items[2].append(words[1].strip())
        elif tag == "date":
            if item.text is not None:
                items[3].append(item.text.strip())
            else:
                items[3].append("")



    return items

def main(argv):

    # usage
    if len(argv)<2:
        print("Usage: python3 %s <metadata dir>", argv[0])
        exit()

    dir = argv[1]

    # fp_1 = open("../data/metadata_first.csv","w")
    fp_1 = open("/var/tmp/xzheng20_mhe_cs365_final/data_final/process/metadata_first.csv", "w")
    fp_1.write("id,creator,type,material,date\n")

    # fp_2 = open("../data/metadata_full.csv","w")
    fp_2 = open("/var/tmp/xzheng20_mhe_cs365_final/data_final/process/metadata_full.csv", "w")
    item_nums = [12,15,29,3]
    list = ["id"]
    for i in range(item_nums[0]):
        list.append("creator"+str(i))
    for i in range(item_nums[1]):
        list.append("type"+str(i))
    for i in range(item_nums[2]):
        list.append("material"+str(i))
    for i in range(item_nums[3]):
        list.append("date"+str(i))
    fp_2.write(",".join(list)+"\n")

    # size = []

    # read the directory
    filelist = readdir(dir)

    for file in filelist:
        id = file.split("/")[-1][:7]
        metadata = parseXML(file)
        for item in metadata:
            if len(item)==0:
                item.append("")

        fp_1.write(id+","+metadata[0][0]+","+metadata[1][0]+","+metadata[2][0]+","+metadata[3][0]+"\n")

        list = [id]
        for i in range(4):
            list.extend(metadata[i])
            list.extend([""]*(item_nums[i]-len(metadata[i])))
        fp_2.write(",".join(list)+"\n")

        # print(len(metadata[0]),len(metadata[1]),len(metadata[2]),len(metadata[3]))
        # size.append([len(metadata[0]),len(metadata[1]),len(metadata[2]),len(metadata[3])])

    fp_1.close()
    fp_2.close()

    # npsize = np.matrix(size)
    # print(np.max(npsize,axis=0))


if __name__ == "__main__":
    main(sys.argv)

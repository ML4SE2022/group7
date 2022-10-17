
def reduceJson():
    try:
        f = open("train_codesearchnet_7.json")
    except:
        print("DOES NOT WORK")
        return
    writeFile = open("shortTrain.json", "w")
    counter = 0
    for i in range(1):
        line = f.readline()  # only one long line
        numSeen, numElementsWanted = 0, 20

        for chr in line:
            counter += 1
            if (chr == "}"):
                numSeen += 1
                if (numSeen == numElementsWanted):
                    break

        print("LINE:", line[:counter], "\n\n")
        writeFile.write(line[:counter]+"]")
        return
        writeFile.write(line)

        counter += 1
        if (counter == 10):
            f.close()
            writeFile.close()
            return
    print("kom hit i stedet")
    f.close()
    writeFile.close()


reduceJson()

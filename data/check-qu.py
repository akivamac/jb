import os

while True:
    print ()
    uqe = input("Type something: ")
    print ()

    stat = os.system(f"grep '{uqe}' conversations.txt")
    grep_returned = os.waitstatus_to_exitcode(stat)

    if grep_returned == 1:
        print ("Not in training! You can add it!")
    elif grep_returned == 0:
        print ("Sorry, already in there:")
        print (grep_returned)
    else:
        print ("⚠️ grep error ⚠️")

def list_split(elems,k):
    num_elems=len(elems)
    outcome=[]
    temp=[]
    for i in range(num_elems):
        if i%k==0 and i!=0:
            outcome.append(temp)
            temp=[]
        temp.append(float(elems[i]))
    outcome.append(temp)
    return outcome
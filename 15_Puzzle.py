
def is_solve(board)
	#olde={4:[0,0],3:[0,1],1:[0,2],2:[0,3]}
	cal=[]
	cost=board[0]
	old=board.items()
#dict_items([(1, [0, 2]), (2, [0, 3]), (3, [0, 1]), (4, [0, 0])])
	look=[[(0,0),1],[(0,1),2],[(0,2),3],[(1,0),4],[(1,1),5],[(1,2),6],[(2,0),4],[(2,1),8],[(2,2),0]]

	for i in look:
		i2=list(i[0])
		for j in old:
			if j[1]==i2:
				cal.append(j[0])
  
	count=0
	for i in cal:
		for j in cal:
			if (i>j and cal.index(i)<cal.index(j)):
            count+=1
	if count%2==0:
		return(board)
	else:
		a=swap(board)
		if is_solve(a):
			return(board)
		else:
			return False

def movement(board):
	original=board.copy()
	ocost=original[0]
	origin=original[1]
    listoflist=[(ocost,origin)]
    count=0
    x,y=origin[0]
    x1=x-1
    while (x1>=0 and count<1):
        for k,v in origin.items():
            if v==[x1,y]:
                origin[k]=[x,y]
                origin[0]=[x1,y]
                x1=x1-1
				if is_goal(origin):
					return(origin)
				else:
					if is_solve(origin):
						cost=heuristic(origin)
						listoflist.append(ocost+cost[0],cost[1])
						count+=1
						break
   
   
	
	original=board.copy()
	ocost=original[0]
	origin=original[1]
    listoflist=[(ocost,origin)]
    count=0
    x,y=origin[0]
    x1=x+1
    while (x1<=2 and count<1):
        for k,v in origin.items():
            if v==[x1,y]:
                origin[k]=[x,y]
                origin[0]=[x1,y]
                x1=x1-1
				if is_goal(origin):
					return(origin)
				else:
					if is_solve(origin):
						cost=heuristic(origin)
						listoflist.append(ocost+cost[0],cost[1])
						count+=1
						break
						
					
					
	original=board.copy()
	ocost=original[0]
	origin=original[1]
    listoflist=[(ocost,origin)]
    count=0
    x,y=origin[0]
    y1=y+1
    while (y1<=2 and count<1):
        for k,v in origin.items():
            if v==[x1,y]:
                origin[k]=[x,y]
                origin[0]=[x,y1]
                y1=y1+1
				if is_goal(origin):
					return(origin)
				else:
					if is_solve(origin):
						cost=heuristic(origin)
						listoflist.append(ocost+cost[0],cost[1])
						count+=1
						break
						


	original=board.copy()
	ocost=original[0]
	origin=original[1]
    listoflist=[(ocost,origin)]
    count=0
    x,y=origin[0]
    y1=y-1
    while (y1>=0 and count<1):
        for k,v in origin.items():
            if v==[x1,y]:
                origin[k]=[x,y]
                origin[0]=[x,y1]
                y1=y1-1
				if is_goal(origin):
					return(origin)
				else:
					if is_solve(origin):
						cost=heuristic(origin)
						listoflist.append(ocost+cost[0],cost[1])
						count+=1
						break
						


board={2: [0, 0],1: [0, 1], 3: [0, 2], 4: [1, 0], 5: [1, 1], 6: [1, 2], 7: [2, 0], 8: [2, 1], 0: [2, 2]}

################################################################################################	
def heuristic(board)
	cost1=0
	pz=board
#goal=[(1,[0,0]),(2,[0,1]),(3,[0,2]),(4,[0,3]),(5,[1,0]),(6,[1,1]),(7,[1,2]),(8,[1,3])]
#pz=[(1,[0,0]),(2,[0,1]),(3,[0,2]),(4,[0,3]),(5,[1,0]),(6,[1,1]),(7,[1,2]),(8,[1,3])]
	mand=0
	for i in pz:
        dgt=i[0]
        x,y=i[1]
        #y=i[2]
        for j in goal:
            if j[0]==dgt:
                nx,ny=j[1]
                #ny=j[2]
                mand+=abs(x-nx)+abs(y-ny)
				cost1+=mand
	#return(cost,pz)
	return(cost,pz)

def is_goal(board):
	if heuristic(board)[1] ==0:
		return True
	else:
		return False

def heuristic(board):
    cost1=0
    pz=board
    goal={1:[0,0],2:[0,1],3:[0,2],4:[1,0],5:[1,1],6:[1,2],7:[2,0],8:[2,1],0:[2,2]}
#pz=[(1,[0,0]),(2,[0,1]),(3,[0,2]),(4,[0,3]),(5,[1,0]),(6,[1,1]),(7,[1,2]),(8,[1,3])]
    mand=0
    #for k,v in pz.items():
    for i in range(0,9):
        dgt=i
        x,y=pz.get(dgt)
        nx,ny=goal.get(dgt)
        if x!=nx or y!=ny:
            mand=abs(x-nx)+abs(y-ny)
            cost1+=mand
        #return(cost,pz)
    return(cost1,pz)
board={2: [0, 0], 1: [0, 1], 3: [0, 2], 4: [1, 0], 5: [1, 1], 6: [1, 2], 7: [2, 0], 8: [2, 1], 0: [2, 2]}
print(heuristic(board))		
		
###########################################################################################
	
	
def solver(initial)
	fringe=[initial]
	while len(fringe)>0:
		s=fringe.pop(0)
		if is_goal(s[1]):
			return(s[1])
		else:
			if is_solve(s[1]):
				fringe.append(movement(s))
				fringe= sorted(fringe,key=lambda x:x[0])
			continue
		


######################################################		
	
def swap(board):
    cord=[[0,0][0,2],[2,0],[2,2]]
    nc=board[0]
    if nc in cord:
        for k,j in board.items():
            if j in cord:
                board[k]=nc
                board[0]=j
                return(board)
    else:
        return False	
	
	
		
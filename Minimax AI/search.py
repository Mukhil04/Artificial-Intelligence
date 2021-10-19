import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
      moveList = []
      moveTree = {}
      return (evaluate(board), moveList, moveTree) 
    if side == False:
      value1 = -10000
      moveTree = {}
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value2, arr, dictionary = minimax(newside, newboard, newflags, depth - 1)
        moveTree[encode(move[0], move[1], move[2])] = dictionary
        if value2 > value1:
          value1 = value2
          move1 = move
          moveList = arr
      moveList.insert(0, move1)
      return (value1, moveList, moveTree)
    elif side == True:
      value1 = 10000
      moveTree = {}
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value2, arr, dictionary = minimax(newside, newboard, newflags, depth - 1)
        moveTree[encode(move[0], move[1], move[2])] = dictionary
        if value2 < value1:
          value1 = value2
          move1 = move
          moveList = arr
      moveList.insert(0, move1)
      return (value1, moveList, moveTree)


    raise NotImplementedError("you need to write this!")

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
      moveList = []
      moveTree = {}
      return (evaluate(board), moveList, moveTree) 
    if side == False:
      moveTree = {}
      moveList = []
      move1 = [[], [], None]
      value1 = -10000
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value2, arr, dictionary = alphabeta(newside, newboard, newflags, depth - 1, alpha = alpha, beta = beta)
        moveTree[encode(move[0], move[1], move[2])] = dictionary
        if value2 > alpha:
          alpha = value2
        if alpha >= beta:
          return (alpha, [], moveTree)
        else:
          if alpha > value1:
            value1 = alpha
            move1 = move
            moveList = arr
      moveList.insert(0, move1)
      return (value1, moveList, moveTree)
    elif side == True:
      moveTree = {}
      moveList = []
      move1 = [[], [], None]
      value1 = 10000
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value2, arr, dictionary = alphabeta(newside, newboard, newflags, depth - 1, alpha = alpha, beta = beta)
        moveTree[encode(move[0], move[1], move[2])] = dictionary
        if value2 < beta:
          beta = value2
        if beta <= alpha:
          return (beta, [], moveTree)
        else:
          if beta < value1:
            value1 = beta
            move1 = move
            moveList = arr
      moveList.insert(0, move1)
      return (value1, moveList, moveTree)

    raise NotImplementedError("you need to write this!")
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    count = 0
    dictionary = {}
    moveLists = {}
    moveTree_big = {}
    d = depth-1
    for move in generateMoves(side, board, flags):
      newside1, newboard1, newflags1 = makeMove(side, board, move[0], move[1], flags, move[2])
      leaf_node = []
      dic = {}
      for i in range(breadth):
        newside = newside1
        newboard = newboard1
        newflags = newflags1
        depth = d
        if i == 0:
          move1 = []
        arr = []
        while depth != 0:
          value, moveList, moveTree = random(newside, newboard, newflags, chooser)
          arr.append([moveTree, encode(moveList[0][0], moveList[0][1], moveList[0][2])])
          if len(moveList) == 0:
            leaf_node.append(value)
            continue
          if i == 0:
            move1.insert(0, moveList[0])
          newside, newboard, newflags = makeMove(newside, newboard, moveList[0][0], moveList[0][1], newflags, moveList[0][2])
          depth = depth - 1
        for item in range(len(arr), -1, -1):
          if item == len(arr) or item == len(arr) - 1:
            continue
          else:
            arr[item][0][arr[item][1]] = arr[item+1][0]
        dic[arr[0][1]] = arr[0][0][arr[0][1]]
        leaf_node.append(value)
      moveTree_big[encode(move[0], move[1], move[2])] = dic
      dictionary[encode(move[0], move[1], move[2])] = sum(leaf_node) /breadth
      moveLists[encode(move[0], move[1], move[2])] = move1
    if side:
      val = min (list(dictionary.values()))
    else:
      val = max(list(dictionary.values()))
    position = list(dictionary.values()).index(val)
    final = list(dictionary.keys())[position]
    moveLists[final].insert(0, decode(final))
    move_final = moveLists[final]
    return (val, move_final, moveTree_big)
    raise NotImplementedError("you need to write this!")

from collections import deque
from queue import Queue
from enum import IntEnum

class Func:
    SQRT = 'sqrt'
    EXP  = 'exp'

    @classmethod
    def isinstance(cls, instance):
        return instance == cls.SQRT or \
               instance == cls.EXP

class Paren:
    OPEN  = '('
    CLOSE = ')'

    @classmethod
    def isinstance(cls, instance):
        return instance == cls.OPEN or \
               instance == cls.CLOSE


class Op:
    PLUS    = '+'
    MUL     = '*'
    DIV     = '/'
    MINUS   = '-'
    POWER   = '^'

    @classmethod
    def isinstance(cls, instance):
        return instance == cls.PLUS     or \
               instance == cls.MINUS    or \
               instance == cls.MUL      or \
               instance == cls.DIV      or \
               instance == cls.POWER
    

class OpPrecedence:
    """Defind all ops and precedece
    Ref: https://en.cppreference.com/w/c/language/operator_precedence"""
    precedence = {
        Op.POWER    : 4,
        Op.PLUS     : 2,
        Op.MINUS    : 2,
        Op.MUL      : 3,
        Op.DIV      : 3
    }
     
    @classmethod
    def get(cls, op):
        return cls.precedence[op]


class OpAssociativity:
    """Defind all ops and associativity
    Ref: https://en.cppreference.com/w/c/language/operator_precedence"""
    associativity = {
        Op.POWER    : 1, # right-to-left
        Op.PLUS     : 0, # left-to-right
        Op.MINUS    : 0,
        Op.MUL      : 0,
        Op.DIV      : 0
    }

    @classmethod
    def get(cls, op):
        return cls.associativity[op]


def to_decimal(sequence):
    """Convert each element to decimal if possible"""
    ret = []
    for e in sequence:
        try:
            e = int(e)
        except ValueError:
            try:
                e = float(e)
            except ValueError:
                pass
        finally:
            ret.append(e)
    return ret


def valid_parentheses(sequence):
    """Return True if a sequence contains valid parentheses"""
    stack = deque()

    for e in sequence:
        if e == Paren.OPEN:
            stack.append(e)
        if e == Paren.CLOSE:
            try:
                if stack.pop() != Paren.OPEN:
                    return False
            except IndexError:
                return False

    return len(stack) == 0


def infix_to_prefix(sequence):
    """Convert infix expression to prefix expression"""
    assert valid_parentheses(sequence), 'Invalid parentheses'

    output_stack = deque()
    op_stack = deque()

    for token in sequence[::-1]:
        if isinstance(token, (int, float)):
            output_stack.append(token)
        elif Func.isinstance(token):
            op_stack.append(token)
        else:
            if Op.isinstance(token):
                # pop the operator stack onto output stack if it should
                while len(op_stack) > 0 and \
                    ((Func.isinstance(op_stack[-1])) or \
                    (Op.isinstance(op_stack[-1]) 
                        and OpPrecedence.get(token) < 
                            OpPrecedence.get(op_stack[-1])) or \
                    (Op.isinstance(op_stack[-1])
                        and OpPrecedence.get(token) == 
                            OpPrecedence.get(op_stack[-1])
                        and OpAssociativity.get(token) == 1)) and \
                    (op_stack[-1] != Paren.CLOSE):

                    output_stack.append(op_stack.pop())
                
                # push onto operator stack
                op_stack.append(token)

            if token == Paren.CLOSE:
                op_stack.append(token)

            if token == Paren.OPEN:
                while len(op_stack) > 0 and \
                    (op_stack[-1] != Paren.CLOSE):
                    output_stack.append(op_stack.pop())
                if len(op_stack) > 0 and op_stack[-1] == Paren.CLOSE:
                    op_stack.pop()

    # pop remaining the operator stack onto output stack
    while len(op_stack):
        output_stack.append(op_stack.pop())

    # format return
    ret = []
    while len(output_stack):
        ret.append(output_stack.pop())

    return ret


def infix_to_postfix(sequence):
    """Convert infix expression to postfix expression"""
    assert valid_parentheses(sequence), 'Invalid parentheses'

    output_queue = Queue()
    op_stack = deque()

    # scan left-to-right
    for token in sequence:
        if isinstance(token, (int, float)):
            output_queue.put(token)
        elif Func.isinstance(token):
            op_stack.append(token)
        else:
            if Op.isinstance(token):
                # pop the operator stack onto output queue if it should
                while len(op_stack) > 0 and \
                    ((Func.isinstance(op_stack[-1])) or \
                    (Op.isinstance(op_stack[-1]) 
                        and OpPrecedence.get(token) < 
                            OpPrecedence.get(op_stack[-1])) or \
                    (Op.isinstance(op_stack[-1])
                        and OpPrecedence.get(token) == 
                            OpPrecedence.get(op_stack[-1])
                        and OpAssociativity.get(token) == 0)) and \
                    (op_stack[-1] != Paren.OPEN):

                    output_queue.put(op_stack.pop())
                
                # push onto operator stack
                op_stack.append(token)

            if token == Paren.OPEN:
                op_stack.append(token)

            if token == Paren.CLOSE:
                while len(op_stack) > 0 and \
                    (op_stack[-1] != Paren.OPEN):
                    output_queue.put(op_stack.pop())
                if len(op_stack) > 0 and op_stack[-1] == Paren.OPEN:
                    op_stack.pop()

    # pop remaining the operator stack onto output queue
    while len(op_stack):
        output_queue.put(op_stack.pop())

    # format return
    ret = []
    while not output_queue.empty():
        ret.append(output_queue.get())

    return ret


def standardize_unary(sequence, pos):
    """Standardize -x into (0 - x)
    Args:
        sequence: list of tokens
        pos: index of unary operator
    """
    output_sequence = sequence[:pos]
    output_sequence.extend([Paren.OPEN, 0, Op.MINUS])
    
    # if a number x follows unary operator, convert to (0 - x)
    if isinstance(sequence[pos+1], (int, float)):
        output_sequence.extend([sequence[pos+1], Paren.CLOSE])
        if pos + 2 < len(sequence):
            output_sequence.extend(sequence[pos+2:])
    # if a clause (...) or a function follows unary operator,
    # convert to (0 - (...)) 
    elif Func.isinstance(sequence[pos+1]) or sequence[pos+1] == Paren.OPEN:
        paren_stack = deque()
        j = pos + 1
        while j < len(sequence):
            if sequence[j] == Paren.OPEN:
                paren_stack.append(Paren.OPEN)
            if sequence[j] == Paren.CLOSE:
                assert paren_stack.pop() == Paren.OPEN, 'Invalid parentheses'
                if len(paren_stack) == 0:
                    break
            j += 1

        # complete editing output sequence
        output_sequence.extend(sequence[pos+1:j+1] + [Paren.CLOSE] + sequence[j+1:])

    return output_sequence


def unary_to_binary(sequence):
    # recursively check and standardize unary operator

    if sequence[0] == Op.MINUS:
        output_sequence = standardize_unary(sequence, 0)
        # recursively check until there is no unary
        return unary_to_binary(output_sequence)

    output_sequence = []
    unary = True

    for i, e in enumerate(sequence):
        if e == Op.MINUS:
            unary = True
            p = i - 1
            while p > 0 and \
                not Op.isinstance(sequence[p]) and \
                not Func.isinstance(sequence[p]) and \
                sequence[p] != Paren.OPEN:
                if isinstance(sequence[p], (int, float)) or sequence[p] == Paren.CLOSE:
                    unary = False
                    break
                p -= 1

            if unary:
                output_sequence = standardize_unary(sequence, i)
                # recursively check until there is no unary
                return unary_to_binary(output_sequence)
        
        output_sequence.append(e)
    
    return output_sequence
    

if __name__ == '__main__':
    eqs = ['2 * ( - 4 + 3 ^ 2 )', '2 * 4 ^ 3 ^ 2', '2 * 4 ^ 3 ^ sqrt ( 2 )']
    postfix_sol = [
        [2, 0, 4, '-', 3, 2, '^', '+', '*'], 
        [2, 4, 3, 2, '^', '^', '*'],
        [2, 4, 3, 2, 'sqrt', '^', '^', '*']
    ]
    prefix_sol = [
        ['*', 2, '+', '-', 0, 4, '^', 3, 2],
        ['*', 2, '^', 4, '^', 3, 2],
        ['*', 2, '^', 4, '^', 3, 'sqrt', 2]
    ]

    for i, eq in enumerate(eqs):
        sequence = to_decimal(eq.split())
        sequence = unary_to_binary(sequence)
        assert infix_to_postfix(sequence) == postfix_sol[i], print(infix_to_postfix(sequence))
        assert infix_to_prefix(sequence) == prefix_sol[i]

    print('Test pass')

    # demo
    eq = 'exp ( 2 ) * 12 ^ 3 + 4 / ( 3 * sqrt ( 2 ) )'
    sequence = to_decimal(eq.split())
    sequence = unary_to_binary(sequence)
    print('')
    print('{:<12s}{:<12s}'.format('Infix: ', eq))
    print('{:<12s}{:<12s}'.format('Prefix: ', str(infix_to_prefix(sequence))))
    print('{:<12s}{:<12s}'.format('Postfix: ', str(infix_to_postfix(sequence))))
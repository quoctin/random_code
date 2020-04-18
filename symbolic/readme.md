# Conversion from Infix expression to Prefix expression and Postfix expression

Usual mathematical expression (`infix`) represents addition of `a` and `b` as `a + b`. More complicatedly, parentheses `(...)` are required to remove any abiguity or specify a preference, for instance: `2 * ( ( 4 + 3)  ^ 2 )`.

[Polish notation (PN)](https://en.wikipedia.org/wiki/Polish_notation) is a more efficient representation, where parentheses are not needed. In PN, the operators preceed their operands. For instance, `2 * ( ( 4 + 3)  ^ 2 )` can be written in PN as `* 2 ^ + 4 3 2`.

[Reverse Polish notation (RPN)](https://en.wikipedia.org/wiki/Reverse_Polish_notation),  also known as Polish postfix notation,  is a more efficient representation, where parentheses are not needed. In RPN, the operators follow their operands. For instance, `2 * ( ( 4 + 3)  ^ 2 )` can be written in RPN as `2 4 3 + 2 ^ *`.

In this Python [script]('eq_representation.py'), I implemented **Shunting-yard algorithm** that converts from infix to prefix and postfix expression. My implemention also allows unary operation (`-x`).

# Example

```
Infix:      ['exp', '(', 2, ')', '*', 12, '^', 3, '+', 4, '/', '(', 3, '*', 'sqrt', '(', 2, ')', ')']
Prefix:     ['+', '*', 'exp', 2, '^', 12, 3, '/', 4, '*', 3, 'sqrt', 2]
Postfix:    [2, 'exp', 12, 3, '^', '*', 4, 3, 2, 'sqrt', '*', '/', '+']
```
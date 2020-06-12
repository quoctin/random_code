import queue

class Node:
    NUL = 'NUL'
    def __init__(self, value, cate=None, child=None, parent=None, suffix=None, output=None):
        self.value = value
        self.cate = cate if cate else []
        self.child = child if child else []
        self.parent = parent
        self.suffix = suffix
        self.output = output

    def add_cate(self, cate):
        self.cate.append(cate)

    def add_child(self, node):
        self.child.append(node)
        node.parent=self

    
class Graph:
    def __init__(self):
        self.root = Node(Node.NUL)

    def add_word(self, node: Node, word: str):
        if not word: 
            return node
        element = word[0]
        for c in node.child:
            if c.value == element:
                return self.add_word(c, word[1:]) if 1 < len(word) else c
        new_node = Node(element)
        node.add_child(new_node)
        return self.add_word(new_node, word[1:]) if 1 < len(word) else new_node

    def add_suffix_output_links(self):
        q = queue.Queue()
        q.put(self.root)
        while q.qsize() > 0:
            node = q.get()
            if node.value == Node.NUL:
                node.suffix = None 
            elif node.parent.value == Node.NUL:
                node.suffix = node.parent 
            else:
                psuffix = node.parent.suffix
                while not node.suffix:
                    for c in psuffix.child:
                        if node.value == c.value:
                            node.suffix = c
                            node.output = c if c.cate else c.output
                            break
                    else:
                        if psuffix.value == Node.NUL:
                            node.suffix = psuffix
                            node.output = c if c.cate else c.output
                            break
                        else:
                            psuffix = psuffix.suffix
            for c in node.child:
                q.put(c)

    def build(self, categories: dict):
        for cate, word_list in categories.items():
            for word in word_list:
                node = self.add_word(self.root, word)
                node.add_cate(cate)
        self.add_suffix_output_links()

    def search(self, sequence: str):
        def format_output(output: list):
            ret = []
            for node in output:
                sequence = [node.value]
                parent = node.parent
                while parent:
                    if parent.value != Node.NUL:
                        sequence.append(parent.value)
                    parent = parent.parent 
                ret.append((node.cate, ''.join(sequence[::-1])))
            return ret

        output = []
        node = self.root 
        for s in sequence:
            found = False
            while not found and node:
                for c in node.child:
                    if c.value == s:
                        node = c
                        if node.cate:
                            output.append(node)
                        spread = node
                        while spread.output:
                            if spread.output.cate:
                                output.append(spread.output)
                            spread = spread.output
                        found = True
                        break # found then break
                else:
                    if node.value == Node.NUL:
                        break # found nothing, move to next character
                    else:
                        node = node.suffix
        return format_output(output)

if __name__ == "__main__":
    categories = {
        1: ['ab', 'bcc'],
        2: ['ba', 'baa', 'aa']
    }

    g = Graph()
    g.build(categories)

    # TEST
    assert g.root.value == Node.NUL, 'Fail'
    assert g.root.child[0].value == 'a', 'Fail'
    assert g.root.child[1].value == 'b', 'Fail'
    assert g.root.child[0].child[0].value == 'b', 'Fail'
    assert g.root.child[1].child[0].value == 'c', 'Fail'
    assert g.root.child[1].child[1].value == 'a', 'Fail'
    assert g.root.value == Node.NUL, 'Fail'
    assert g.root.child[0].child[0].cate == [1], 'Fail'
    assert g.root.child[1].child[0].child[0].cate == [1], 'Fail'
    assert g.root.child[1].child[1].child[0].cate == [2], 'Fail'
    print('Pass')

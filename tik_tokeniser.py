import tiktoken

tokeniser = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <ENDOFTEXT> In the sunlit terraces of someunknownPlace")

integers = tokeniser.encode(text, allowed_special={"<ENDOFTEXT>"})

print(integers)
print(tokeniser.decode(integers))

text = ("Akwirw ier")

integers = tokeniser.encode(text, allowed_special={"<ENDOFTEXT>"})

print(integers)
print(tokeniser.decode(integers))


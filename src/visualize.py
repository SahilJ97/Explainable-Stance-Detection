header = """
<head>
<style>
.zero {
  opacity: 0.0;
}
.one {
  opacity: 0.1; 
}
.two {
  opacity: 0.2; 
}
.three {
  opacity: 0.3; 
}
.four {
  opacity: 0.4; 
}
.five {
  opacity: 0.5; 
}
.six {
  opacity: 0.6; 
}
.seven {
  opacity: 0.7; 
}
.eight {
  opacity: 0.8; 
}
.nine {
  opacity: 0.9; 
}
.ten {
  opacity: 1.0; 
}

</style>
</head>
"""

weight_to_class = {
    .0: "zero",
    .1: "one",
    .2: "two",
    .3: "three",
    .4: "four",
    .5: "five",
    .6: "six",
    .7: "seven",
    .8: "eight",
    .9: "nine",
    1.: "ten",
}

EPSILON = 1e-6


def get_words_html(words, weights):
    denom = max(weights)
    out_html = []
    for word, weight in zip(words, weights):
        rounded = round(float(weight/(denom + EPSILON)), 1)
        opacity = weight_to_class[rounded]
        out_html.append(f"<span class=\"{opacity}\" display: inline>{word}</span>")
    return " ".join(out_html)


if __name__ == "__main__":
    print(weight_to_class.keys())
    print(header)
    text, weights = ["hello", "world"], [.55, 1.2]
    print(get_words_html(text, weights))
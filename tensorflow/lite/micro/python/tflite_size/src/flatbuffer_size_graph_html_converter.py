# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

HTML_HEAD = """
<!DOCTYPE html>
<!-- reference: https://www.w3schools.com/howto/howto_js_treeview.asp -->
<html>
<head>
<style>
/* Remove default bullets */
ul {
  list-style-type: none;
}


/* Style the caret/arrow */
.caret {
  cursor: pointer;
  user-select: none; /* Prevent text selection */
}

/* Create the caret/arrow with a unicode, and style it */
.caret::before {
  content: "\\25B6";
  color: black;
  display: inline-block;
  margin-right: 6px;
}

/* Rotate the caret/arrow icon when clicked on (using JavaScript) */
.caret-down::before {
  transform: rotate(90deg);
}

/* Hide the nested list */
.nested {
  display: none;
}

/* Show the nested list when the user clicks on the caret/arrow (with JavaScript) */
.active {
  display: block;
}
</style>



</head>
<body>
<ul id="root">
"""

HTML_TAIL = """
</ul>

<script>
    var toggler = document.getElementsByClassName("caret");
    var i;

    for (i = 0; i < toggler.length; i++) {
      toggler[i].addEventListener("click", function() {
        this.parentElement.querySelector(".nested").classList.toggle("active");
        this.classList.toggle("caret-down");
        });
    }    
</script>

</body>
</html>

"""


class HtmlConverter:
  """ A class to convert the size graph to a tree of collapsible list """

  def __init__(self):
    self._html_body = HTML_HEAD

  def _draw_collapsible_list(self, node):
    if node.isLeaf is True or len(node.children) == 0:
      self._html_body += "<li> %s: %s (size: %d) </li>\n" % (
          node.name, node.value, node.size)
    else:
      self._html_body += "<li> <span class = \"caret\"> %s (size: %d) </span>\n" % (
          node.name, node.size)
      self._html_body += "<ul class=\"nested\">\n"
      for node in node.children:
        self._draw_collapsible_list(node)
      self._html_body += "</ul>\n"
      self._html_body += "</li>\n"

  def display_flatbuffer(self, root):
    self._html_body = HTML_HEAD
    self._draw_collapsible_list(root)
    self._html_body += HTML_TAIL
    return self._html_body

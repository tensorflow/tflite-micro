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
    self._htmlBody = HTML_HEAD

  def _drawCollapsibleList(self, fNode):
    #print("visiting %s %d %d " % (fNode.name , fNode.isLeaf, len(fNode.children)))
    if fNode.isLeaf is True or len(fNode.children) == 0:
      self._htmlBody += "<li> %s: %s (size: %d) </li>\n" % (
          fNode.name, fNode.value, fNode.size)
    else:
      self._htmlBody += "<li> <span class = \"caret\"> %s (%d) </span>\n" % (
          fNode.name, fNode.size)
      self._htmlBody += "<ul class=\"nested\">\n"
      for node in fNode.children:
        self._drawCollapsibleList(node)
      self._htmlBody += "</ul>\n"
      self._htmlBody += "</li>\n"

  def displayFlatbuffer(self, rootNode):
    self._htmlBody = HTML_HEAD
    self._drawCollapsibleList(rootNode)
    self._htmlBody += HTML_TAIL
    return self._htmlBody

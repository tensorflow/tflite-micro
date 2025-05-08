var strgURL =   location.pathname;                      // path of current component

// constructor for the array of objects
function tabElement(id, folderName, tabTxt )  {
  this.id = id;                                       // elementID as needed in html;
  this.folderName = folderName;                       // folder name of the component
  this.tabTxt = tabTxt;                               // Text displayed as menu on the web
  this.currentListItem = '<li id="' + this.id + '" class="current"> <a href="../' + this.folderName + '/index.html"><span>' + this.tabTxt + '</span></a></li>';
  this.listItem = '<li id="' + this.id + '"> <a href="../' + this.folderName + '/index.html"><span>' + this.tabTxt + '</span></a></li>';
};

// constructor for the array of objects
function tabSubElement(id, folderName, tabTxt )  {
  this.id = id;                                       // elementID as needed in html;
  this.folderName = folderName;                       // folder name of the component
  this.tabTxt = tabTxt;                               // Text displayed as menu on the web
  this.currentListItem = '<li id="' + this.id + '" class="current"> <a href="../../' + this.folderName + '/index.html"><span>' + this.tabTxt + '</span></a></li>';
  this.listItem = '<li id="' + this.id + '"> <a href="../' + this.folderName + '/index.html"><span>' + this.tabTxt + '</span></a></li>';
};

// array of main tabs
var arr = [];

// fill array
 arr.push( new tabElement( "GEN",     "General",     "Overview"));
 arr.push( new tabElement( "CORE",    "Core",        "Core"));
 arr.push( new tabElement( "DRV",     "Driver",      "Driver"));
 arr.push( new tabElement( "RTOS2",   "RTOS2",       "RTOS2"));
 arr.push( new tabElement( "DSP",     "DSP",         "DSP"));
 arr.push( new tabElement( "NN",      "NN",          "NN"));
 arr.push( new tabElement( "View",    "View",        "View"));
 arr.push( new tabElement( "Compiler","Compiler",    "Compiler"));
 arr.push( new tabElement( "Toolbox", "Toolbox",     "Toolbox"));
 arr.push( new tabElement( "Stream",  "Stream",      "Stream"));
 arr.push( new tabElement( "DAP",     "DAP",         "DAP"));
 arr.push( new tabElement( "Zone",    "Zone",        "Zone"));
 
// array of sub tabs fore Core
var arr_sub = [];
 arr_sub.push( new tabSubElement( "CORE_M",  "Core",     "Cortex-M"));
 arr_sub.push( new tabSubElement( "CORE_A",  "Core_A",   "Cortex-A"));

// write main tabs
// called from the header file
function writeComponentTabs()  {
  for ( var i=0; i < arr.length; i++ ) {
    str = "/" + arr[i].folderName + "/"
    if (strgURL.search(str) > 0) {                    // if this is the current folder
      document.write(arr[i].currentListItem);         // then print and highlight the tab
    } else {
      // specially for Core_A need to highlight the Core tab too
      if ((arr[i].id=="CORE") && (strgURL.search("/Core_A/")>0)){
        document.write(arr[i].currentListItem);         // then print and highlight the tab
      } else {
         document.write(arr[i].listItem);                // else, print the tab
      }
    }
  }
};

// write sub-tabs (can't use layout XML as will show all other sub-tabs as well (API, usage, et.))
// called from the header file
function writeSubComponentTabs()  {
  if((strgURL.search("/Core/")>0)||(strgURL.search("/Core_A/")>0)){
    document.write('<div id="navrow1" class="tabs">');
    document.write('<ul class="tablist">');
    for ( var i=0; i < arr_sub.length; i++ ) {
      str = "/" + arr_sub[i].folderName + "/"
      if (strgURL.search(str) > 0) {                    // if this is the current folder
          document.write(arr_sub[i].currentListItem);         // then print and highlight the tab
       } else {
        document.write(arr_sub[i].listItem);                              // else, print the tab
      }
    }
    document.write('</ul>');
    document.write('</div>');
  }
};

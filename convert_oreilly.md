# Prep for O'Reilly submission

<https://docs.atlas.oreilly.com/writing_in_asciidoc.html>

## Change the "pkg" CSS class to be just **bold**

in TMwR.css? This is not working yet for me

## Generate `.md` files:

Choose a directory to put the new files in (use `_bookdown.yml` to generate only part of the book):

```r
library(bookdown)
render_book(output_format = html_book(keep_md = TRUE), 
  output_dir = "tmwr-to-ch9/")
```

## Convert divs to markdown images

In new directory:

```
sed -i ".bak" 's/<p class=\"caption\">\(.*\)<\/p>/STARTCAP\1STOPCAP/g' *.md
sed -i ".bak" 's/<div class=\"figure\" style="text-align: center">//g' *.md
sed -i ".bak" 's/<img src=\"\(.*\)\" alt=.*/STARTIMAGE\1STOPIMAGE/g' *.md
perl -i~ -0777 -pe 's/STARTIMAGE(.*?)STOPIMAGE\nSTARTCAP\(\\#fig\:(.*?)\)(.*?)STOPCAP\n<\/div>/[[\2]]\n![\3](\1)/g' *.md
```

## Convert to asciidoc using pandoc

In the new directory:

```
for f in *.md; do pandoc --markdown-headings=atx \
    --verbose \
    --wrap=none \
    --reference-links \
    --citeproc \
    --bibliography=TMwR.bib \
    --lua-filter=lower-header.lua \
    -f markdown -t asciidoc \
    -o "${f%.md}.adoc" \
    "$f"; done
```

## Fix notes/warnings

Using sed:

```
sed -i ".bak" "s/:::rmdnote/[NOTE]\n====/g" *.adoc   
sed -i ".bak" "s/:::rmdwarning/[WARNING]\n====/g" *.adoc   
sed -i ".bak" "s/:::/====/g" *.adoc
sed -i ".bak" -E "s/^{empty}//g" *.adoc
sed -i ".bak" -E "1 s/\[#([^()]*)]*\]/\[\1\]/" *.adoc
sed -i ".bak" -E "s/\@ref\(fig:([^()]*)\)/<<\1>>/g" *.adoc
sed -i ".bak" -E "s/\@ref\(tab:([^()]*)\)/<<\1>>/g" *.adoc
sed -i ".bak" -E "s/\@ref\(([^()]*)\)/<<\1>>/g" *.adoc
perl -i~ -0777 -pe 's/\[\[refs\]\].*\Z//sg' *.adoc
perl -i~ -0777 -pe 's/\.\(\#tab\:(.*?)\)(.*?)/[[\1]]\n\.\2/g' *.adoc
sed -i ".bak" 's/\[\[\(.*\)\]\] image:\(.*\)\[\(.*\)\]/\[\[\1\]\]\n\.\3\nimage::\2/g' *.adoc
sed -i ".bak" 's/Figure <</<</g' *.adoc
sed -i ".bak" 's/Table <</<</g' *.adoc
```

## Make preface [actually a preface](https://docs.atlas.oreilly.com/writing_in_asciidoc.html#prefaces-PntlujUD)

Make beginning make sense, remove front matter, and also a preface. In the new directory (then edit):

```
mv index.adoc preface.adoc
emacs preface.adoc
```

## Convert SVG to PNG in /premade dir

```
mogrify -format png *.svg
```

## Clean up extra files when totally done

```
rm *.bak
rm *.md
rm *.html
rm *~
rm premade/*.svg
rm -r libs
```

## Zip up and send!



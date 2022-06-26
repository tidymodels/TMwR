# Prep for O'Reilly submission

<https://docs.atlas.oreilly.com/writing_in_asciidoc.html>

## Change the "pkg" CSS class to be just **bold**

in TMwR.css? This is not working yet for me

## Generate `.md` files:

Choose a directory to put the new files in (use `_bookdown.yml` to generate only part of the book):

```r
library(bookdown)
render_book(output_format = html_book(keep_md = TRUE), 
  output_dir = "tmwr-atlas/")
```

## Convert divs to markdown images

In new directory:

```
sed -i ".bak" 's/<p class=\"caption\">\(.*\)<\/p>/STARTCAP\1STOPCAP/g' *.md
sed -i ".bak" 's/<div class=\"figure\" style="text-align: center">//g' *.md
sed -i ".bak" 's/<img src=\"\(.*\)\" alt=.*/STARTIMAGE\1STOPIMAGE/g' *.md
perl -i~ -0777 -pe 's/STARTIMAGE(.*?)STOPIMAGE\nSTARTCAP\(\\#fig\:(.*?)\)(.*?)STOPCAP\n<\/div>/[[\2]]\n![\3](\1)/g' *.md
sed -i ".bak" "s/:::rmdnote/STARTNOTE/g" *.md  
sed -i ".bak" "s/:::rmdwarning/STARTWARNING/g" *.md
sed -i ".bak" "s/:::/STOPBOX/g" *.md
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

## Fix notes/warnings/image/etc

Using sed:

```
sed -i ".bak" "s/STARTNOTE/[NOTE]\n====\n/g" *.adoc   
sed -i ".bak" "s/STARTWARNING/[WARNING]\n====\n/g" *.adoc   
sed -i ".bak" "s/STOPBOX/\n====/g" *.adoc
sed -i ".bak" -E "s/^{empty}//g" *.adoc
sed -i ".bak" -E "1 s/\[#([^()]*)]*\]/\[\1\]/" *.adoc
sed -i ".bak" -E "s/\@ref\(fig:([^()]*)\)/<<\1>>/g" *.adoc
sed -i ".bak" -E "s/\@ref\(tab:([^()]*)\)/<<\1>>/g" *.adoc
sed -i ".bak" -E "s/\@ref\(([^()]*)\)/<<\1>>/g" *.adoc
perl -i~ -0777 -pe 's/\[\[refs\]\].*\Z//sg' *.adoc
perl -i~ -0777 -pe 's/\.\(\#tab\:(.*?)\)(.*?)/[[\1]]\n\.\2/g' *.adoc
sed -i ".bak" 's/\[\[\(.*\)\]\] image:\(.*\)\[\(.*\)\]/\[\[\1\]\]\n\.\3\nimage::\2\[\]/g' *.adoc
sed -i ".bak" 's/image::figures/image::images/g' *.adoc
sed -i ".bak" 's/image::premade/image::images/g' *.adoc
sed -i ".bak" 's/\.svg/\.png/g' *.adoc
sed -i ".bak" 's/Figure <</<</g' *.adoc
sed -i ".bak" 's/Table <</<</g' *.adoc
sed -i ".bak" 's/Chapters <</<</g' *.adoc
sed -i ".bak" 's/Chapter <</<</g' *.adoc
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
ren -v "*.adoc" "#1.asciidoc"
mv pre-proc-table.asciidoc appendix.asciidoc
rm *.bak
rm *.html
rm *~
rm -r libs
ren -v "*-*.asciidoc" "ch#1.asciidoc"
```

## Zip up and send!



# Prep for O'Reilly submission

<https://docs.atlas.oreilly.com/writing_in_asciidoc.html>

## Change the "pkg" CSS class to be just **bold**

in TMwR.css? This is not working yet for me

## Generate `.md` files:

Choose a directory to put the new files in (use `_bookdown.yml` to generate only part of the book):

```r
render_book(output_format = html_book(
  keep_md = TRUE, pandoc_args = "--lua-filter=lower-header.lua"), 
  output_dir = "files_for_print/")
```

We don't need the HTML files so `rm *.html` in the new directory

## Convert to asciidoc using kramdown: <https://github.com/asciidoctor/kramdown-asciidoc>

In the new directory:

```
find ./ -name "*.md" \
    -type f \
    -exec sh -c \
    'kramdoc {}' \;
```

```
pandoc --markdown-headings=atx \
    --verbose \
    --wrap=none \
    --reference-links \
    --citeproc \
    --bibliography=TMwR.bib \
    --lua-filter=lower-header.lua \
    -f markdown -t asciidoc \
    -o 01-software-modeling.adoc \
    01-software-modeling.md
```

## Fix notes/warnings

Using sed:

```
sed -i ".bak" "s/:::rmdnote/[NOTE]\n====/g" *.adoc   
sed -i ".bak" "s/:::rmdwarning/[WARNING]\n====/g" *.adoc   
sed -i ".bak" "s/:::/====/g" *.adoc
sed -i ".bak" -E "1 s/\[#([^()]*)]*\]/\[\1\]/" *.adoc
sed -i ".bak" -E "s/\\\@ref\(fig:([^()]*)\)/<<\1>>/g" *.adoc
sed -i ".bak" -E "s/\\\@ref\(tab:([^()]*)\)/<<\1>>/g" *.adoc
sed -i ".bak" -E "s/\\\@ref\(([^()]*)\)/<<\1>>/g" *.adoc
```

## Make some changes to `index.adoc`

Make beginning make sense and also a preface

```
mv index.adoc preface.adoc  
```

## Clean up extra files when totally done

```
rm *.bak
rm *.md
```

## Zip up and send!



# Writing Documentation

When adding new features to garage, it is important that these features are
well-documented and accessible. Garage's high-level docs contain information on
workflows and processes that are regularly used by developers and end-users.They
are home to various how-to's and examples that help others use garage
effectively.

High-level documentation should focus on describing such workflows, or, in the
case of newly-added features, describing how these features might be used when
running experiments. For code-level documentation such as docstrings and style
guides, please see the [CONTRIBUTING.md](https://github.com/rlworkgroup/garage/blob/master/CONTRIBUTING.md).

For general documentation best practices, [this](https://www.writethedocs.org/guide/)
is a great guide that covers a broad range of topics. You may also reach out to
us should you have specific questions.

## Where to Put Documentation

You can choose to update the existing docs under the `docs/user` directory, or
create a new page dedicated to the subject you are documenting. Be sure to
update the table of contents in `docs/index.md` to reflect your changes. Your
doc will most likely go under one of the existing headers. This page, for
example, is listed under `Development Guides` by adding
`user/writing_documentation` to that section:

```
.. toctree::
   :maxdepth: 2
   :caption: Development Guides

   user/testing
   user/benchmarking
   user/writing_documentation
```

## General Guidelines

We suggest going through the documentation to familiarize yourself with how docs
are written in garage. We've included a list of do's and don'ts to keep in mind
when writing documentation:

### Do's

- Keep your doc self-contained - avoid having users go through multiple pages to
  understand how to use a feature or become familiar with a workflow.

- Make your doc accessible - Include examples to help demonstrate a point or
  feature, and clarify code snippets by providing context.

- Make your doc readable - Make use of markdown styling (see [this page]([https://commonmark.org/help/](https://commonmark.org/help/))
  for a markdown cheatsheet) - see the section below for more on styling.

- Refer to external sources - There are several high-quality online resources
  that explain various reinforcement learning techniques. We refer users to them
  all the time, you should too!

### Don'ts

- Claim other work as your own. Please cite external sources - you may use
  footnotes for this purpose.

## Styling

You can take advantage of styling to make your documentation readable. We
recommend you go through the markdown cheatsheet linked above for a complete
list of the styles and formatting options available to you. Here, we'll
emphasize the use of two important ones:

### Code Blocks

Code blocks and syntax highlighting are effective and simple to use. They
highlight code snippets in your doc and make them easy to identify. Use three
bat ticks "`" to signify the opening and closing of a code block, and append the
code language you're using to the opening bat ticks to enable syntax
highlighting:

````
```python
def very_readable(int arg):
```
````

Results in:

```python
def very_readable(int arg):
```

You can also use one bat tick for inline code snippets, class references, or
directory paths:

``you can `cd`  into `~/garage/data/` ``

you can `cd` into `~/garage/data/`

### Sphinx

Our documentation pages are built with Sphinx, which means you can render more
complicated shapes within text. Garage docstrings mostly use the Sphinx `math`
directive to render shapes and symbols in math equations. This is also possible
in markdown:

````
```math
E = m c^2
```
````

```math
E = m c^2
```

This is made possible with Recommonmark.
Their [documentation](https://readthedocs.org/projects/recommonmark/downloads/pdf/latest/)
contains an extensive list of what can by done with Sphinx in markdown.
[This](https://github.com/rlworkgroup/garage/blob/master/CONTRIBUTING.md#docstrings)
section in the `CONTRIBUTING.md` also contains
various examples that demonstrate how the math directive should be used.

We specifically use this when specifying shapes of input and output tensors in
docstrings (and you should too), but we've included this here incase you need
to specify tensor shapes in your doc.

### Viewing Your Doc

You should build and render garage's documentation locally before submitting
a PR to verify that your changes appear as intended. You'll need to have the
developer dependencies installed (via `garage[dev]`, see [this page](installation.html))
for this to work.  To do this, from the garage root directory, run:

```
make docs
```

Once the build process is complete, open the `docs/_build/html/index.html`, as
well as other files you potentially created, to view the render.

After you've finalized your documentation, submit a pull request and the garage
maintainers will review it. Once your PR has been approved, it will be merged
into the master branch and available for end-users to read.

*This page was authored by Mishari Aliesa ([@maliesa96](https://github.com/maliesa96)).*

---
layout: archive
title: ""
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

<style>
  /* Wraps the iframe and handles centering */
  .pdf-wrapper{
    display:flex;                 /* enable flexbox */
    justify-content:center;       /* center horizontally */
    align-items:center;           /* center vertically */
    min-height:100vh;             /* fill viewport height */
    padding:5vh 0;                /* equal 5 vh margin top & bottom */
    box-sizing:border-box;
  }

  /* The iframe itself */
  .pdf-wrapper iframe{
    width:100%;                   /* full width of wrapper */
    height:90vh;                  /* 90 % of viewport height (5 vh padding each side) */
    border:none;
  }

  /* Slightly shorter on very small screens */
  @media (max-width:600px){
    .pdf-wrapper iframe{
      height:85vh;
    }
  }
</style>

<div class="pdf-wrapper">
  <iframe
    src="{{ site.baseurl }}/files/Yuxuan_Resume.pdf#toolbar=0&navpanes=0&scrollbar=0"
    loading="lazy">
  </iframe>
</div>




<!-- Education
======
* Ph.D in Version Control Theory, GitHub University, 2018 (expected)
* M.S. in Jekyll, GitHub University, 2014
* B.S. in GitHub, GitHub University, 2012

Work experience
======
* Spring 2024: Academic Pages Collaborator
  * Github University
  * Duties includes: Updates and improvements to template
  * Supervisor: The Users

* Fall 2015: Research Assistant
  * Github University
  * Duties included: Merging pull requests
  * Supervisor: Professor Hub

* Summer 2015: Research Assistant
  * Github University
  * Duties included: Tagging issues
  * Supervisor: Professor Git
  
Skills
======
* Skill 1
* Skill 2
  * Sub-skill 2.1
  * Sub-skill 2.2
  * Sub-skill 2.3
* Skill 3

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul> -->
  
<!-- Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Currently signed in to 43 different slack teams -->

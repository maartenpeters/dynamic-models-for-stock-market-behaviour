% Bestand: uva-inf-presentatie.cls
% Auteur:  Frederick Kreuk (r.f.a.c.kreuk@uva.nl)
%          origineel door Stephen Swatman
%
% Dit bestand is de studentenvariant van het presentatietemplate dat binnen
% Academische Vaardigheden Informatica 1 en 2 wordt gebruikt. Deze studentenvariant
% is uitgevoerd in blauw in plaats van het tijdens bijeenkomsten gebruikte paars.
%
% AVI slides voor het nieuwe decennium redesign van juli 2020 door Frederick Kreuk
%
% Opties:
%
%   sidebar                 Voeg een sidebar toe met inhoudsopgave.
%   dyslexic                Gebruik het OpenDyslexic-lettertype dat de leesbaarheid
%                           voor dyslectische studenten potentieel verbeterd.
%   classic                 gebruik de lay-out van het oude template van voor 2020

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uva-inf-presentation}

\RequirePackage{translations}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\RequirePackage{arrayjobx}
\RequirePackage[percent]{overpic}
\RequirePackage{tikz}
% \RequirePackage[utf8x]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{booktabs}

% uva-inf-common
%-------------------------------------------------------------------------------
\DeclareTranslation{dutch}{internalSupervisor}{Interne Supervisor}
\DeclareTranslation{english}{internalSupervisor}{Internal Supervisor}

\DeclareTranslation{dutch}{mentor}{Mentor}
\DeclareTranslation{english}{mentor}{Mentor}

\DeclareTranslation{dutch}{of}{van}
\DeclareTranslation{english}{of}{of}

\DeclareTranslation{dutch}{externalSupervisor}{Externe Supervisor}
\DeclareTranslation{english}{externalSupervisor}{External Supervisor}

\DeclareTranslation{dutch}{class}{Programma}
\DeclareTranslation{english}{class}{Programme}

\DeclareTranslation{dutch}{student}{Student}
\DeclareTranslation{english}{student}{Student}

\DeclareTranslation{dutch}{students}{Studenten}
\DeclareTranslation{english}{students}{Students}

\DeclareTranslation{dutch}{course}{Cursus}
\DeclareTranslation{english}{course}{Course}

\DeclareTranslation{dutch}{course_id}{Vakcode}
\DeclareTranslation{english}{course_id}{Course code}

\DeclareTranslation{dutch}{uva_logo}{logoUvA_nl}
\DeclareTranslation{english}{uva_logo}{logoUvA_en}

\DeclareTranslation{dutch}{uva_logo_white}{logoUvA_nl_white}
\DeclareTranslation{english}{uva_logo_white}{logoUvA_en_white}

\newcommand*{\assignment}[1]{\gdef\@assignment{#1}}
\newcommand*{\@assignment}{$\backslash assignment\{\}$ is unset}

\newcommand*{\assignmenttype}[1]{\gdef\@assignmenttype{#1}}
\newcommand*{\@assignmenttype}{$\backslash assignmenttype\{\}$ is unset}

\newcommand*{\internalSupervisor}[1]{\gdef\@internalSupervisor{#1}}
\newcommand*{\@internalSupervisor}{}

\newcommand*{\mentor}[1]{\gdef\@mentor{#1}}
\newcommand*{\@mentor}{}

\newcommand*{\class}[1]{\gdef\@class{#1}}
\newcommand*{\@class}{}

\newcommand*{\externalSupervisor}[1]{\gdef\@externalSupervisor{#1}}
\newcommand*{\@externalSupervisor}{}

\newcommand*{\programme}[1]{\gdef\@programme{#1}}
\newcommand*{\@programme}{}

\newcommand*{\course}[1]{\gdef\@course{#1}}
\newcommand*{\@course}{}

\newcommand*{\courseid}[1]{\gdef\@courseid{#1}}
\newcommand*{\@courseid}{}

\newcounter{authorcount}
\setcounter{authorcount}{0}

\newcounter{uvacount}
\setcounter{uvacount}{0}

\newcommand\setprefix[2]{\expandafter\def\csname#1\endcsname{#2}}
\newcommand\getprefix[1]{\csname#1\endcsname}

\NewDocumentCommand\authors{>{\SplitList{;}}m}{
    \ProcessList{#1}{ \addauthor }
}
\newcommand\addauthor[1]{
    \stepcounter{authorcount}
    \setprefix{author\arabic{authorcount}}{#1}
}
\renewcommand{\author}[1]{
	\addauthor{#1}
}
\NewDocumentCommand\uvanetids{>{\SplitList{;}}m}{
    \ProcessList{#1}{ \adduvanetid }
}
\newcommand\adduvanetid[1]{
    \stepcounter{uvacount}
    \setprefix{uva\arabic{uvacount}}{#1}
}
\newcommand\uvanetid[1]{
	\adduvanetid{#1}
}

%-------------------------------------------------------------------------------

% Teken een andreaskruis. Nu met transparantie voor additioneel profijt!
\newcommand{\andreas}[5][1.0]{
    \begin{scope}[shift={(#2,#3)},rotate=45,scale=#4,transparency group,opacity=#1]
        \fill[fill=#5] (0,0) -- (0,1) -- (3,1) -- (3,0);
        \fill[fill=#5] (1,-1) -- (2,-1) -- (2,2) -- (1,2);
    \end{scope}
}

% Pad naar logo's
\graphicspath{{./logo/}}

% Geef opties door aan de beamer klasse. Gaat voornamelijk om de aspect ratio.
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}

% Registreer of de gebruiker een sidebar wil.
\DeclareOption{sidebar}{\newcommand{\addsidebar}{}}

% Registreer of de gebruiker het OpenDyslexic-lettertype wil gebruiken.
\DeclareOption{dyslexic}{\newcommand{\useopendyslexic}{}}

% Registreer of de gebruiker het OpenDyslexic-lettertype wil gebruiken.
\DeclareOption{classic}{\newcommand{\useTraditionalAppearance}{}}

% Process de opties en laad de beamer klasse.
\ProcessOptions
\LoadClass[14pt]{beamer}

% Stel default thema in (fabulous).
\ifdefined\useTraditionalAppearance
\else
    \ifdefined\useModernAppearance
    \else
        \providecommand{\useFabulousColours}
    \fi
\fi

% Bij fabulous kleuren hoort het moderne thema.
\ifdefined\useFabulousColours
    \providecommand{\useModernAppearance}
\fi

% Gebruikt voor het tekenen van gradienten en vormen.
\RequirePackage{tikz}
\usetikzlibrary{fadings,calc,positioning,fit,backgrounds}
\RequirePackage{adjustbox}
\RequirePackage{varwidth}
\RequirePackage{calc}
\RequirePackage{listofitems}
\RequirePackage{xstring}
\RequirePackage{tcolorbox}
\tcbuselibrary{skins,breakable}

% Zodat we sinuswaarden kunnen berekenen voor de transparantie van de kruisjes.
\RequirePackage{xfp}

%-------------------------------------------------------------------------------
% Colors
\definecolor{uvaRed}{RGB}{188,0,49}
\definecolor{uvaBlack}{RGB}{31,29,33}
\definecolor{uvaGrey15}{RGB}{220,221,222}
\definecolor{uvaGrey8}{RGB}{235,235,236}

% Definitie kleuren.
\definecolor{studentblauw}{RGB}{27,78,117}

\definecolor{bengelblauw_licht}{RGB}{20,70,139}
\definecolor{bengelblauw_donker}{RGB}{14,25,70}

% Op sommige dagen heb je een regenboog nodig.
\pgfdeclareverticalshading{rainbow}{100bp}
{color(0bp)=(red); color(25bp)=(red); color(35bp)=(yellow);
color(45bp)=(green); color(55bp)=(cyan); color(65bp)=(blue);
color(75bp)=(violet); color(100bp)=(violet)}

% Cirkelvormige schaduw (gebruikt voor sidebar en glanseffect frametitel)
\tikzfading[name=fade out,
    inner color=transparent!80,
    outer color=transparent!100]

% Verticale glans (gebruikt voor glans- en schaduweffect op rand sidebar)
\tikzfading[name=vertical fade,
    top color=transparent!100,
    bottom color=transparent!100,
    middle color=transparent!90]

% Horizontale glans (gebruikt voor glans- en schaduweffect onder frametitel)
\tikzfading[name=horizontal simple fade,
    left color=transparent!80,
    right color=transparent!100]
\tikzfading[name=horizontal fade,
    left color=transparent!100,
    right color=transparent!100,
    middle color=transparent!80]

% Highlight de huidige slide in de sidebar. De code is absoluut walgelijk.
\newcommand{\highlight}[1]{%
\tcbox[nobeforeafter,tcbox width=auto limited,enhanced,left=1pt,right=1pt,top=1pt,bottom=1pt,boxsep=0pt,arc=2pt,
boxrule=0pt,opacityfill=0.2,colback=white,coltext=uvaBlack,
grow to left by=1pt,grow to right by=1pt,enlarge top by=1pt,enlarge bottom by=-1pt,halign=flush left,width=2.2cm,parbox=false]{#1}
}

% Stel achtergrondkleuren in.
\newcommand{\backgroundA}{uvaGrey8}
\newcommand{\backgroundB}{uvaGrey15}

\ifdefined\useFabulousColours
    \renewcommand{\backgroundA}{uvaGrey8}
    \renewcommand{\backgroundB}{uvaGrey15}
\fi

\ifdefined\useTraditionalAppearance
    \renewcommand{\backgroundA}{uvaGrey8}
    \renewcommand{\backgroundB}{uvaGrey15}
\fi

% Negeer blok wanneer optie appearanceDraft is gegeven.
\newcommand{\draftblock}[1]{\ifdefined\isDraft\else#1\fi}

% Voer blok uit indien het nieuwe thema wordt gebruikt.
\newcommand{\ifmodern}[1]{\ifdefined\useModernAppearance#1\fi}
% Voer eerste blok uit indien het nieuwe thema wordt gebruikt,
% voer het tweede blok uit indien de optie appearanceTraditional is gegeven.
\newcommand{\ifmodernelse}[2]{\ifdefined\useModernAppearance#1\else#2\fi}

\newcommand{\setlinespacing}{\renewcommand{\baselinestretch}{1}}

% Als daarom gevraagd wordt, moeten we OpenDyslexic maar instellen als lettertype.
% Nu ook in soort-van-vetgedrukt en soort-van-schuingedrukt!
\ifxetex
    \ifdefined\useopendyslexic
        \RequirePackage{fontspec}
        \setsansfont{OpenDyslexic}
        \renewcommand{\setlinespacing}{\renewcommand{\baselinestretch}{1.2}}
    \fi
\else
    \RequirePackage[utf8]{inputenc}
    \RequirePackage[T1]{fontenc}
\fi

% We hoeven geen lijn boven de footnotes.
\renewcommand\footnoterule{\relax}

% Deze code wordt uitgevoerd om de sidebar te maken.
\ifdefined\addsidebar

% Laad de sidebar.
\useoutertheme[width=2.5cm,height=0cm,right]{sidebar}

% Maak een omgeving zonder sidebar voor de titelpagina.
\newenvironment{titelframe}{
    \advance\textwidth\beamer@sidebarwidth\begin{frame}
}{
    \end{frame}
}

% Thema van de sidebar.
\setbeamertemplate{sidebar canvas \beamer@sidebarside}{
    \ifmodern{\draftblock{
    % De hele achtergrond is een prachtige tikz tekening, welke de illusie
    % wekt dat de sidebar zich bevindt op een laag achter de rest van de slide.
    \begin{tikzpicture}[remember picture,x=1pt,y=1pt]
        % Glans op rand.
        \fill[white,path fading=vertical fade,draw=none,anchor=north]
            (-0.5pt,0) rectangle (0,\paperheight);
        % Scherpe schaduw onder rand.
        \fill[uvaBlack,path fading=vertical fade,draw=none,anchor=north]
            (0,0) rectangle (0.5pt,\paperheight);
        % Vage schaduw die aangeeft dat sidebar onder de content ligt.
        \clip (0,0) rectangle (\beamer@sidebarwidth,\paperheight);
        \fill[uvaBlack,path fading=fade out,draw=none,anchor=north]
            (0,0.5*\paperheight) circle [x radius=\beamer@sidebarwidth,
                                         y radius=0.8\paperheight];
    \end{tikzpicture}
    }}
}
\setbeamertemplate{sidebar \beamer@sidebarside}{
    \vspace{3mm}
    \hskip 1.7pt
    % Print de titel van de presentatie.
    {
    \begin{beamercolorbox}[wd=\beamer@sidebarwidth]{title}
        \usebeamerfont{title in sidebar}\usebeamercolor[fg]{title in sidebar}
        \inserttitle\par
    \end{beamercolorbox}
    
    }
    
    \hskip 1pt

    % Dit maakt de daadwerkelijke navigatie.
    \setlinespacing
    \ifmodernelse{
    % Plaats in adjustbox om te voorkomen dat in een lange bijeenkomst
    % de inhoud over de paginanummering heen gekotst wordt.
    \begin{adjustbox}{
        clip=true,
        max height=0.85\paperheight
    }
        \insertverticalnavigation{\beamer@sidebarwidth}
    \end{adjustbox}
    }{\insertverticalnavigation{\beamer@sidebarwidth}}
    \renewcommand{\baselinestretch}{1}
}
\else

% Zonder sidebar hoeft de titelomgeving niets te doen!
\newenvironment{titelframe}{\begin{frame}}{\end{frame}}
\fi

% De achtergrond van de hele presentatie.
\setbeamertemplate{background}{
    \ifmodernelse{
    % In het moderne thema is de achtergrond een gradient, gedecoreerd met een
    % transparant wit Andreaskruis.
    \begin{tikzpicture}[remember picture]
        \fill[shading=axis,left color=\backgroundA,right color=\backgroundB,
                shading angle=135] (0,0)
                rectangle (\the\paperwidth,\the\paperheight);
        % Teken een + en draai het 45 graden om een Andreaskruis te maken.
        \draftblock{
        \begin{scope}[shift={(0.859375\paperwidth,0)},rotate=45,scale=2,transparency group,opacity=0.2]
            \fill[uvaGrey15] (0,0) -- (0,1) -- (3,1) -- (3,0);
            \fill[uvaGrey15] (1,-1) -- (2,-1) -- (2,2) -- (1,2);
        \end{scope}
        }
    \end{tikzpicture}
    }{
    % In het traditionele thema is de achtergrond FNWI-paars met een witte balk
    % aan de onderkant, gedecoreerd met een groot zwart Andreaskruis.
    \begin{tikzpicture}
        % Vul de achtergrond eerst met kleur, tot net voor de onderkant.
        \fill[color=\backgroundA] (0,0.08\paperheight)
            rectangle (\the\paperwidth,\the\paperheight);
        % Teken een + en draai het 45 graden om een Andreaskruis te maken.
        \begin{scope}[shift={(0.859375\paperwidth,0)},rotate=45,scale=2]
            \fill[uvaGrey15] (0,0) -- (0,1) -- (3,1) -- (3,0);
            \fill[uvaGrey15] (1,-1) -- (2,-1) -- (2,2) -- (1,2);
        \end{scope}
        % De balk aan de onderkant moet wit zijn.
        \fill[color=white] (0,0) rectangle (\the\paperwidth,0.08\paperheight);
    \end{tikzpicture}
    }
}

% De stijl van de titelpagina.
\setbeamertemplate{title page}{
    \vfill

    \begin{beamercolorbox}{title}\centering
		{\Large\bfseries\inserttitle}\vspace{0.5cm}
        % We kunnen de kleine kruisjes natuurlijk niet vergeten.
    \draftblock{
    \begin{beamercolorbox}{title}\centering\begin{tikzpicture}
        \newcommand\numcross{3}
        \foreach \n in {1,...,\the\numexpr\numcross\relax} {
            \andreas[\fpeval{0.9*sin((\n + 1)*pi/(\numcross+2))}]{0.5*\n}{0}{0.12}{uvaRed}
        }
    \end{tikzpicture}\end{beamercolorbox}
    }\\[0.5cm]
	    \@date\\[0.4cm]
    \end{beamercolorbox}

    \begin{beamercolorbox}{title}
	\begin{minipage}[t]{0.4\paperwidth}
	    \strut\vspace*{-\baselineskip}
        \footnotesize
		\begin{flushleft}
			\textbf{\ifnum\value{authorcount}>1\relax
                    \GetTranslation{students}%
                  \else
                    \GetTranslation{student}%
                  \fi:}\\
          \foreach \index in {1, ..., \value{authorcount}} {%
              {\getprefix{author\index}\\ {\getprefix{uva\index}\\[0.2cm]}}
          }
		\end{flushleft}
	\end{minipage}
    \hfill
	\begin{minipage}[t]{0.4\paperwidth}
	    \strut\vspace*{-\baselineskip}
        \footnotesize
		\begin{flushright}
		    \begin{tabular}{@{}l@{}}
    			\ifdefempty{\@internalSupervisor}{}{
        			\textbf{\GetTranslation{internalSupervisor}:} \\
        			\@internalSupervisor \\[0.2cm]
    			}
    			\ifdefempty{\@externalSupervisor}{}{
        			\textbf{\GetTranslation{externalSupervisor}:} \\
        			\@externalSupervisor \\[0.2cm]
        		}
    			\ifdefempty{\@mentor}{}{
        			\textbf{\GetTranslation{mentor}:} \\
        			\@mentor \\[0.2cm]
    			}
    			\ifdefempty{\@class}{}{
        			\textbf{\GetTranslation{class}:} \\
        			\@class \\[0.2cm]
        		}
        	\end{tabular}
		\end{flushright}
	\end{minipage}
	\end{beamercolorbox}
    \vfill
}

% De stijl van de titel van elke pagina.
\setbeamertemplate{frametitle}{
    \ifmodernelse{
        \vskip0.1cm
        \begin{beamercolorbox}[ht=1cm]{frametitle}
            % Om de titel visueel te scheiden van de inhoud, worden achter de tekst verscheidene gradients
            % getekend om de illusie te wekken dat de titel zich bevindt op een verhoogd stuk van de achtergrond.
            % Deze loopt aan de rechterkant vloeiend over in de achtergrond.
            \draftblock{\begin{tikzpicture}[overlay,remember picture,x=\textwidth,y=1pt]
                % Definitie bounds van titelscheur.
                \node (upperLeft) at ( $ (-1cm,0) +(0,-8pt) $ ) {};
                \node (upperRight) at ( $ (\textwidth+0.5cm,0) +(0,-8.5pt) $ ) {};
                \node (lowerLeft) at ( $ (-1cm,0) +(0,-8.5pt) $ ) {};
                \node (lowerRight) at ( $ (\textwidth,0) +(0,-9pt) $ ) {};
                % De onderkant van een scheur is shiny, met daaronder een scherpe schaduw.
                \fill[white,path fading=horizontal simple fade,draw=none,anchor=north] (upperLeft) rectangle (upperRight);
                \fill[uvaBlack,path fading=horizontal simple fade,draw=none,anchor=north] (lowerLeft) rectangle (lowerRight);
                % Highlight van titelscheur.
                \begin{scope}
                    \clip (-2cm,-8.5pt) rectangle (\textwidth,\paperheight);
                    \fill[white,path fading=fade out,draw=none,opacity=0.2]
                        (2cm,-8.5pt) circle [x radius=0.5\textwidth,y radius=2cm];
                \end{scope}
                % Schaduw onder titlescheur/
                \begin{scope}
                    \clip (-2cm,-8.5pt) rectangle (\textwidth,-\paperheight);
                    \fill[uvaBlack,path fading=fade out,draw=none,opacity=0.2]
                        (2cm,-8.5pt) circle [x radius=0.5\textwidth,y radius=0.5cm];
                \end{scope}
            \end{tikzpicture}
            }\begin{adjustbox}{clip=true,max width=0.95\textwidth}\begin{varwidth}{2\textwidth}\insertframetitle\end{varwidth}\end{adjustbox}\hfill
        \end{beamercolorbox}
    }{
        \vskip0.2cm
        \begin{beamercolorbox}[ht=1.2cm,center]{frametitle}
        \insertframetitle
        \end{beamercolorbox}
    }
}

% De stijl van de footer onderin.
\setbeamertemplate{footline}{
    \ifmodernelse{
        % Stel aan dat alles nu horizontaal moet.
        \makebox[\paperwidth][r]{
            % Eerste blok bevat het UvA logo.
            \begin{beamercolorbox}[wd=.45\paperwidth,dp=.3cm,ht=0cm]{footer}
            % Minipage om het logo en onderschrift te managen.
            \begin{minipage}{.45\paperwidth}
            \hspace{0pt}
            \vspace{-1.4mm}
            \vbox{
                \begin{tikzpicture}
                    \node[opacity=0.5,text width=\textwidth] at (0,0) {
                        \includegraphics[height=0.05208\paperheight]
                            {\GetTranslation{uva_logo_white}}};
                    \ifdefempty{\@programme}{}{
                        \node[opacity=0.5,align=left,text width=\textwidth] at (0.6,-0.1) {\fontfamily{lmss}\selectfont{\textsc{\scriptsize \@programme}}\hfill};
                    }
                \end{tikzpicture}
                \vspace{0.0222222\paperheight}
                \hspace{-0.478\paperheight}
            }
            \end{minipage}
            \end{beamercolorbox}
            \hfill
            % Klein blokje voor de paginabenummering.
            \begin{beamercolorbox}[wd=2.5cm,dp=.3cm,ht=0cm]{footer}
                \centering
                \ifdefempty{\@course}{}{
                    \textsc{\@course}
                }
                \textsc{\insertframenumber~\GetTranslation{of} \inserttotalframenumber}
            \end{beamercolorbox}
        }
    }{
        \usebeamercolor[fg]{footer}
        % Eerste blok bevat het UvA logo.
        \begin{minipage}[c][0.05\paperheight]{.4\paperwidth}
        \hspace*{0.012\linewidth}\vspace{-4.5mm}
        \vbox{
            \begin{tikzpicture}
                \node[opacity=1,text width=15cm] at (0,0) {
                    \includegraphics[height=0.05208\paperheight]
                        {\GetTranslation{uva_logo}}};
                \ifdefempty{\@programme}{}{
                    \node[opacity=1,align=left,text width=15cm] at (0.6,-0.1) {\fontfamily{lmss}\selectfont{\textsc{\scriptsize \@programme}}\hfill};
                }
            \end{tikzpicture}
            \vspace{0.0222222\paperheight}
            \hspace{-0.478\paperheight}
        }
        % \begin{overpic}[height=0.05\paperheight]{\GetTranslation{uva_logo}}
        % \ifdefempty{\@programme}{}{
        %     \put (12.5,0) {\fontfamily{lmss}\selectfont\textsc{\scriptsize \@programme}}
        % }
        % \end{overpic}
        \end{minipage}
        \hfill
        \begin{minipage}[c][0.08\paperheight]{.1\paperwidth}
        \centering
        \textsc{\insertframenumber~\GetTranslation{of} \inserttotalframenumber}
        \end{minipage}
        \hfill
        \begin{minipage}[c][0.08\paperheight]{.4\paperwidth}
        \raggedleft
        \textsc{\@course}
        \end{minipage}
        \hspace*{0.02\linewidth}
    }
}

% Secties in de navigatie waar we nu zijn worden gehighlight.
\setbeamertemplate{subsection in sidebar}{
    \draftblock{\beamer@sidebarformat{3.5pt}{subsection in sidebar}{\highlight{\insertsubsectionhead}}}
}


% Elke section krijgt ook een page.
\setbeamertemplate{section page}{
    \vfill
    \begin{beamercolorbox}{title}
        \centering\usebeamerfont{title}\insertsection
    \end{beamercolorbox}
    \vfill
}

% Stel de kleuren in. In het klassieke thema is alle tekst buiten de
% footer wit. In het moderne thema is alle tekst wit.
\setbeamercolor{normal text}{fg=uvaBlack}
\setbeamercolor{title}{fg=uvaBlack}
\setbeamercolor{frametitle}{fg=uvaBlack}
\setbeamercolor{footer}{fg=uvaBlack}
\setbeamercolor{structure}{fg=uvaBlack}
\ifmodern{
    \setbeamercolor{footer}{fg=uvaBlack}
}

\AtBeginEnvironment{example}{
    \setbeamercolor{block title example}{fg=uvaBlack}}

% Het soort bullets dat we willen gebruiken.
\setbeamertemplate{itemize items}[circle]
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize subitem}{\color{uvaBlack}--}

\setcounter{tocdepth}{1}
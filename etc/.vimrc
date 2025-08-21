" ====== General ======
set nocompatible
set encoding=utf-8
"set number
set ruler
set showcmd
set nocursorline
set termguicolors
set background=dark
syntax on
filetype plugin indent on

" Tabs & Indent (기본)
set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
set smartindent

" C/C++ 계열 인덴트(올맨 스타일 선호: 여는 중괄호 개행 후 같은 깊이)
set cindent
" 브레이스와 case 들여쓰기 보수적으로
set cinoptions={0,}0,^0,:0

" 검색/편의
set hlsearch
set incsearch
set ignorecase
set smartcase
set nowrap
set mouse=
set belloff=all

" 색상 테마(내장 중 안정적)
" alpha@AlphaX:~/projects/alpha_cuda$ ls /usr/share/vim/vim82/colors/
" blue.vim      delek.vim    evening.vim   pablo.vim    shine.vim      torte.vim
" darkblue.vim  desert.vim   industry.vim  morning.vim  peachpuff.vim  slate.vim  zellner.vim
" default.vim   elflord.vim  koehler.vim   murphy.vim   ron.vim

colorscheme murphy


" 공백 표시/트레일링 하이라이트
set list
set listchars=tab:»·,trail:·,extends:…,precedes:…
highlight ExtraWhitespace ctermbg=red guibg=#803333
autocmd ColorScheme * highlight ExtraWhitespace ctermbg=red guibg=#803333
autocmd BufWinEnter * match ExtraWhitespace /\s\+$/
autocmd InsertEnter * match ExtraWhitespace /\s\+\%#\@<!$/
autocmd InsertLeave * match ExtraWhitespace /\s\+$/

" ====== Filetype-specific ======
augroup AlphaFiletypes
  autocmd!
  " CUDA 파일을 C++로 하이라이트 (플러그인 없이 안정적으로)
  autocmd BufNewFile,BufRead *.cu,*.cuh setfiletype cpp

  " C/C++/CUDA: 탭/인덴트 4, 스페이스 탭
  autocmd FileType c,cpp,objc,objcpp,cpphdr,cppobj,make setlocal tabstop=4 shiftwidth=4 expandtab cindent

  " CMake: 파일타입 보장 + 인덴트
  autocmd BufNewFile,BufRead CMakeLists.txt,*.cmake setfiletype cmake
  autocmd FileType cmake setlocal tabstop=2 shiftwidth=2 expandtab

  " SQL
  autocmd FileType sql setlocal tabstop=4 shiftwidth=4 expandtab

  " Makefile은 탭 필수 → 스페이스 금지
  autocmd FileType make setlocal noexpandtab tabstop=8 shiftwidth=8

  " Python/쉘 스크립트 자주 쓰면(옵션)
  autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab
  autocmd FileType sh,bash,zsh setlocal tabstop=2 shiftwidth=2 expandtab
augroup END

" ====== 품질-of-life ======
" 저장 시 자동으로 끝 공백 제거 (Makefile 제외)
autocmd BufWritePre * if &ft != 'make' | %s/\s\+$//e | endif

" 백업/스왑 파일을 프로젝트 밖으로
set directory^=$HOME/.vim/swap//
set backupdir^=$HOME/.vim/backup//
set undodir^=$HOME/.vim/undo//
set undofile
silent! call mkdir($HOME."/.vim/swap", "p")
silent! call mkdir($HOME."/.vim/backup", "p")
silent! call mkdir($HOME."/.vim/undo", "p")


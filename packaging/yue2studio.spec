Name:           yue2studio
Version:        @VERSION@
Release:        1%{?dist}
Summary:        YuE2 Music Studio - offline AI song generation (audio.cpp)
License:        MIT
URL:            https://github.com/POWERHACK69/yue2-music-studio
ExclusiveArch:  x86_64

%description
Standalone desktop GUI for YuE2-3B lyrics-conditioned song generation,
powered by the native audio.cpp inference engine (audiocpp_cli).
Models download on first run from inside the app.

%install
mkdir -p %{buildroot}/opt/Yue2Studio
cp -a %{srcdir}/dist/Yue2Studio/* %{buildroot}/opt/Yue2Studio/
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/share/applications
mkdir -p %{buildroot}/usr/share/icons/hicolor/256x256/apps
ln -s /opt/Yue2Studio/Yue2Studio %{buildroot}/usr/bin/Yue2Studio
install -m644 %{srcdir}/packaging/Yue2Studio.desktop %{buildroot}/usr/share/applications/
install -m644 %{srcdir}/app/icon.png %{buildroot}/usr/share/icons/hicolor/256x256/apps/Yue2Studio.png

%files
/opt/Yue2Studio/
/usr/bin/Yue2Studio
/usr/share/applications/Yue2Studio.desktop
/usr/share/icons/hicolor/256x256/apps/Yue2Studio.png

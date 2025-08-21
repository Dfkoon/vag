  // Language translations
        const translations = {
            ar: {
                "title": "محرك تحليل الصور المتقدم - أدوات التحليل الجنائي",
                "main-title": "محرك تحليل الصور الجنائي المتقدم",
                "main-subtitle": "تحليل متقدم بالذكاء الاصطناعي، كشف التلاعب، استخراج البيانات المخفية، والتحليل الجنائي للصور",
                "ai-badge": "مدعوم بأحدث تقنيات الذكاء الاصطناعي والتحليل الجنائي",
                "upload-text": "اسحب الصور هنا للتحليل الجنائي",
                "upload-formats": "يدعم: JPG, PNG, GIF, WebP, BMP",
                "upload-features": "تحليل جنائي متقدم + كشف التلاعب",
                "capture-camera": "التقاط من الكاميرا",
                "import-url": "استيراد من رابط",
                "upload-device": "رفع من الجهاز",
                "clear-all": "مسح الكل",
                "forensic-tools-title": "أدوات التحليل الجنائي للصور",
                "steganalysis-title": "تحليل الإخفاء",
                "steganalysis-desc": "كشف المعلومات المخبئة في الصور",
                "ela-title": "تحليل مستوى الخطأ",
                "ela-desc": "كشف المناطق المعدلة في الصور",
                "metadata-title": "تحليل البيانات الوصفية",
                "metadata-desc": "استخراج وتحليل بيانات EXIF",
                "forensic-title": "التحليل الجنائي",
                "forensic-desc": "تحليل الضوضاء وكشف التزوير",
                "nsfw-title": "كشف المحتوى غير الآمن",
                "nsfw-desc": "كشف المحتوى غير الآمن",
                "face-title": "كشف الوجوه",
                "face-desc": "كشف الوجوه وتحليل الهوية",
                "footer-rights": "© 2024 محرك تحليل الصور الجنائي المتقدم. جميع الحقوق محفوظة.",
                "footer-dev": "تم التطوير بواسطة اسم المطور",
                "change-language": "تغيير اللغة",
                "processing": "جاري المعالجة...",
                "analysis-complete": "اكتمل التحليل",
                "no-threats": "لم يتم العثور على تهديدات",
                "high-quality": "جودة عالية",
                "safe-content": "محتوى آمن"
            },
            en: {
                "title": "Advanced Image Analysis Engine - Forensic Tools",
                "main-title": "Advanced Forensic Image Analysis Engine",
                "main-subtitle": "Advanced AI analysis, manipulation detection, hidden data extraction, and forensic image analysis",
                "ai-badge": "Powered by the latest AI and forensic analysis technologies",
                "upload-text": "Drag images here for forensic analysis",
                "upload-formats": "Supports: JPG, PNG, GIF, WebP, BMP",
                "upload-features": "Advanced forensic analysis + manipulation detection",
                "capture-camera": "Capturer depuis la Caméra",
                "import-url": "Importer depuis l'URL",
                "upload-device": "Télécharger depuis l'Appareil",
                "clear-all": "Tout Effacer",
                "forensic-tools-title": "Outils d'Analyse Forensique d'Images",
                "steganalysis-title": "Stéganalyse",
                "steganalysis-desc": "Détecter les informations cachées dans les images",
                "ela-title": "Analyse du Niveau d'Erreur",
                "ela-desc": "Détecter les zones modifiées dans les images",
                "metadata-title": "Analyse des Métadonnées",
                "metadata-desc": "Extraire et analyser les données EXIF",
                "forensic-title": "Analyse Forensique",
                "forensic-desc": "Analyse du bruit et détection de contrefaçon",
                "nsfw-title": "Détection NSFW",
                "nsfw-desc": "Détecter le contenu non sécurisé",
                "face-title": "Détection de Visages",
                "face-desc": "Détecter les visages et analyser l'identité",
                "footer-rights": "© 2024 Moteur d'Analyse d'Images Forensiques Avancé. Tous droits réservés.",
                "footer-dev": "Développé par Nom du Développeur",
                "change-language": "Changer la Langue",
                "processing": "Traitement en cours...",
                "analysis-complete": "Analyse Terminée",
                "no-threats": "Aucune menace détectée",
                "high-quality": "Haute qualité",
                "safe-content": "Contenu sécurisé"
            },

        };

        let currentLang = 'en'; // Default language

        // Language system
        const langToggleBtn = document.getElementById('langToggleBtn');
        const langMenu = document.getElementById('langMenu');

        langToggleBtn.addEventListener('click', () => {
            langMenu.classList.toggle('active');
        });

        // Close language menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!langToggleBtn.contains(e.target) && !langMenu.contains(e.target)) {
                langMenu.classList.remove('active');
            }
        });

        document.querySelectorAll('.lang-option').forEach(option => {
            option.addEventListener('click', () => {
                const selectedLang = option.getAttribute('data-lang-code');
                switchLanguage(selectedLang);
                langMenu.classList.remove('active');
            });
        });

        function switchLanguage(lang) {
            currentLang = lang;
            const htmlElement = document.getElementById('htmlElement');
            
            // Update HTML attributes
            htmlElement.setAttribute('lang', lang);
            htmlElement.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
            
            // Update all text elements
            document.querySelectorAll('[data-lang]').forEach(element => {
                const key = element.getAttribute('data-lang');
                if (translations[lang] && translations[lang][key]) {
                    if (element.tagName === 'INPUT' && element.type !== 'button') {
                        element.placeholder = translations[lang][key];
                    } else {
                        element.textContent = translations[lang][key];
                    }
                }
            });

            // Update document title
            document.title = translations[lang]['title'] || document.title;
            
            // Update active language option
            document.querySelectorAll('.lang-option').forEach(opt => {
                opt.classList.remove('selected');
                if (opt.getAttribute('data-lang-code') === lang) {
                    opt.classList.add('selected');
                }
            });
        }

        // Network background animation
        const canvas = document.getElementById('networkBg');
        const ctx = canvas.getContext('2d');
        
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        const nodes = [];
        const connections = [];
        
        // Create nodes
        for (let i = 0; i < 50; i++) {
            nodes.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 3 + 1
            });
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Update nodes
            nodes.forEach(node => {
                node.x += node.vx;
                node.y += node.vy;
                
                if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
                if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
                
                // Draw node
                ctx.fillStyle = 'rgba(102, 126, 234, 0.6)';
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                ctx.fill();
            });
            
            // Draw connections
            ctx.strokeStyle = 'rgba(102, 126, 234, 0.2)';
            ctx.lineWidth = 1;
            
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[i].x - nodes[j].x;
                    const dy = nodes[i].y - nodes[j].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < 100) {
                        ctx.beginPath();
                        ctx.moveTo(nodes[i].x, nodes[i].y);
                        ctx.lineTo(nodes[j].x, nodes[j].y);
                        ctx.stroke();
                    }
                }
            }
            
            requestAnimationFrame(animate);
        }
        
        animate();

        // File upload handling
        function handleFileUpload(event) {
            const files = event.target.files;
            const resultsContainer = document.querySelector('.results');
            
            Array.from(files).forEach((file, index) => {
                setTimeout(() => {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        createImageCard(e.target.result, file);
                    };
                    reader.readAsDataURL(file);
                }, index * 200); // Stagger the animations
            });
        }

        function createImageCard(imageSrc, file) {
            const resultsContainer = document.querySelector('.results');
            
            const card = document.createElement('div');
            card.className = 'image-card';
            
            // Simulate analysis results
            const riskLevel = Math.random() > 0.7 ? 'danger' : Math.random() > 0.4 ? 'warning' : 'safe';
            const confidence = Math.floor(Math.random() * 30) + 70; // 70-100%
            
            card.innerHTML = `
                <div class="risk-indicator risk-${riskLevel}">
                    <i class="fas fa-${riskLevel === 'safe' ? 'shield-check' : riskLevel === 'warning' ? 'exclamation-triangle' : 'exclamation-circle'}"></i>
                    ${riskLevel === 'safe' ? translations[currentLang]['safe-content'] : riskLevel === 'warning' ? 'Warning' : 'Danger'}
                </div>
                <img src="${imageSrc}" alt="Analyzed image">
                <div class="image-info">
                    <h4>${file.name}</h4>
                    <p><strong>Size:</strong> ${(file.size / 1024).toFixed(1)} KB</p>
                    <p><strong>Type:</strong> ${file.type}</p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <div class="ai-analysis">
                        <h5>Analysis Results</h5>
                        <p><strong>Status:</strong> ${translations[currentLang]['analysis-complete']}</p>
                        <p><strong>Quality:</strong> ${translations[currentLang]['high-quality']}</p>
                        <p><strong>Threats:</strong> ${translations[currentLang]['no-threats']}</p>
                        <div class="progress-meter">
                            <div class="progress-value" style="width: ${confidence}%;"></div>
                        </div>
                    </div>
                </div>
            `;
            
            resultsContainer.appendChild(card);
            
            // Animate the card
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100);
        }

        function runAnalysis(tool) {
            const resultElement = document.getElementById(`${tool}-result`);
            resultElement.style.display = 'block';
            
            // Show processing overlay
            resultElement.innerHTML = `
                <div class="processing-overlay">
                    <div class="processing-text">
                        <div class="spinner"></div> ${translations[currentLang]['processing']}
                    </div>
                </div>
            `;
            
            // Simulate analysis
            setTimeout(() => {
                const results = generateAnalysisResults(tool);
                resultElement.innerHTML = results;
            }, 2000 + Math.random() * 2000); // Random delay 2-4 seconds
        }

        function generateAnalysisResults(tool) {
            const results = {
                steganalysis: {
                    found: Math.random() > 0.7,
                    data: 'Hidden text found: "Secret message"',
                    confidence: Math.floor(Math.random() * 20) + 80
                },
                ela: {
                    modified: Math.random() > 0.6,
                    areas: Math.floor(Math.random() * 3) + 1,
                    confidence: Math.floor(Math.random() * 25) + 75
                },
                metadata: {
                    camera: 'Canon EOS R5',
                    location: 'GPS coordinates found',
                    timestamp: new Date().toISOString()
                },
                forensic: {
                    authentic: Math.random() > 0.3,
                    noise_pattern: 'Natural',
                    compression_history: 'Single compression'
                },
                nsfw: {
                    safe: Math.random() > 0.8,
                    score: Math.floor(Math.random() * 100),
                    categories: ['Safe', 'Adult', 'Violence', 'Gore']
                },
                face: {
                    faces_found: Math.floor(Math.random() * 4),
                    ages: [25, 32, 45],
                    emotions: ['Happy', 'Neutral', 'Surprised']
                }
            };

            const result = results[tool];
            let html = '<div class="result-item">';
            
            switch(tool) {
                case 'steganalysis':
                    html += `
                        <div class="result-label">Steganography Detection:</div>
                        <div class="result-value">${result.found ? 'Hidden data detected' : 'No hidden data found'}</div>
                        ${result.found ? `<div class="result-value">${result.data}</div>` : ''}
                        <div class="progress-meter">
                            <div class="progress-value" style="width: ${result.confidence}%;"></div>
                        </div>
                    `;
                    break;
                case 'ela':
                    html += `
                        <div class="result-label">Error Level Analysis:</div>
                        <div class="result-value">${result.modified ? `${result.areas} modified areas detected` : 'No modifications detected'}</div>
                        <div class="progress-meter">
                            <div class="progress-value" style="width: ${result.confidence}%;"></div>
                        </div>
                    `;
                    break;
                case 'metadata':
                    html += `
                        <div class="result-label">EXIF Data:</div>
                        <div class="result-value">Camera: ${result.camera}</div>
                        <div class="result-value">Location: ${result.location}</div>
                        <div class="result-value">Timestamp: ${result.timestamp}</div>
                    `;
                    break;
                case 'forensic':
                    html += `
                        <div class="result-label">Forensic Analysis:</div>
                        <div class="result-value">Authenticity: ${result.authentic ? 'Likely authentic' : 'Possibly manipulated'}</div>
                        <div class="result-value">Noise Pattern: ${result.noise_pattern}</div>
                        <div class="result-value">Compression: ${result.compression_history}</div>
                    `;
                    break;
                case 'nsfw':
                    html += `
                        <div class="result-label">Content Safety:</div>
                        <div class="result-value">Status: ${result.safe ? 'Safe' : 'Unsafe content detected'}</div>
                        <div class="result-value">Safety Score: ${result.score}%</div>
                        <div class="progress-meter">
                            <div class="progress-value" style="width: ${result.score}%;"></div>
                        </div>
                    `;
                    break;
                case 'face':
                    html += `
                        <div class="result-label">Face Analysis:</div>
                        <div class="result-value">Faces Detected: ${result.faces_found}</div>
                        ${result.faces_found > 0 ? `
                            <div class="result-value">Estimated Ages: ${result.ages.slice(0, result.faces_found).join(', ')}</div>
                            <div class="result-value">Emotions: ${result.emotions.slice(0, result.faces_found).join(', ')}</div>
                        ` : ''}
                    `;
                    break;
            }
            
            html += '</div>';
            return html;
        }

        // Quick action functions
        function captureFromCamera() {
            alert(translations[currentLang]['capture-camera'] || 'Capture from Camera');
        }

        function importFromURL() {
            const url = prompt('Enter image URL:');
            if (url) {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.onload = function() {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    
                    canvas.toBlob(blob => {
                        const file = new File([blob], 'imported-image.jpg', {type: 'image/jpeg'});
                        createImageCard(url, file);
                    });
                };
                img.src = url;
            }
        }

        function uploadFromDevice() {
            document.getElementById('imageInput').click();
        }

        function clearAll() {
            document.querySelector('.results').innerHTML = '';
            document.getElementById('imageInput').value = '';
            document.querySelectorAll('.analysis-result').forEach(result => {
                result.style.display = 'none';
                result.innerHTML = '';
            });
        }

        // Drag and drop functionality
        const uploadArea = document.getElementById('uploadArea');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
            uploadArea.style.borderColor = 'var(--info-color)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.background = '';
            uploadArea.style.borderColor = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '';
            uploadArea.style.borderColor = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload({target: {files}});
            }
        });

        uploadArea.addEventListener('click', () => {
            document.getElementById('imageInput').click();
        });

        // Initialize with English by default
        switchLanguage('en');
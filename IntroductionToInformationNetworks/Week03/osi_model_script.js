// OSI Model Animation JavaScript
class OSIModelAnimation {
    constructor() {
        this.currentSlide = 1;
        this.totalSlides = 17;
        this.config = {
            message: "Hi",
            protocol: "smtp",
            encryption: "tls12",
            transport: "tcp",
            sourceIP: "192.168.1.100",
            destIP: "74.125.224.108",
            medium: "ethernet",
            networkSize: "wan"
        };
        this.keyboardDebounce = false;
        
        this.init();
    }

    init() {
        this.safeInitialize();
        this.setupEventListeners();
        this.setupKeyboardNavigation();
        this.initializeTooltips();
    }

    updateConfiguration() {
        try {
            this.config.message = document.getElementById('userMessage')?.value || "Hi";
            this.config.protocol = document.getElementById('protocol')?.value || "smtp";
            this.config.encryption = document.getElementById('encryption')?.value || "tls12";
            this.config.transport = document.getElementById('transport')?.value || "tcp";
            this.config.sourceIP = document.getElementById('sourceIP')?.value || "192.168.1.100";
            this.config.destIP = document.getElementById('destIP')?.value || "74.125.224.108";
            this.config.medium = document.getElementById('medium')?.value || "ethernet";
            this.config.networkSize = document.getElementById('networkSize')?.value || "wan";
            
            this.updateAllDisplays();
        } catch (error) {
            console.error('Configuration update error:', error);
        }
    }

    // Generate binary representation of data
    generateBinaryData(text) {
        return text.split('').map(char => 
            char.charCodeAt(0).toString(2).padStart(8, '0')
        ).join(' ');
    }

    // Generate hexadecimal representation
    generateHexData(text) {
        return text.split('').map(char => 
            char.charCodeAt(0).toString(16).padStart(2, '0').toUpperCase()
        ).join(' ');
    }

    // Generate detailed packet structure visualization
    generatePacketStructure(layer, data) {
        const structures = {
            application: {
                headers: [
                    { name: 'SMTP Command', value: 'DATA', size: '4 bytes' },
                    { name: 'From', value: 'user@company.com', size: '20 bytes' },
                    { name: 'To', value: 'friend@example.com', size: '21 bytes' },
                    { name: 'Subject', value: 'Quick Message', size: '13 bytes' },
                    { name: 'Content-Type', value: 'text/plain', size: '10 bytes' },
                    { name: 'Headers Total', value: '', size: '178 bytes' }
                ],
                payload: data
            },
            presentation: {
                headers: [
                    { name: 'TLS Record Type', value: '0x17 (Application Data)', size: '1 byte' },
                    { name: 'TLS Version', value: '0x0303 (TLS 1.2)', size: '2 bytes' },
                    { name: 'Record Length', value: '0x00C8 (200 bytes)', size: '2 bytes' },
                    { name: 'IV (Initialization Vector)', value: '12-byte random', size: '12 bytes' },
                    { name: 'Auth Tag', value: '16-byte GMAC', size: '16 bytes' }
                ],
                payload: '[ENCRYPTED SMTP DATA]'
            },
            session: {
                headers: [
                    { name: 'Session ID', value: 'SESS_789ABC123', size: '4 bytes' },
                    { name: 'Sequence Number', value: '0x0001', size: '2 bytes' },
                    { name: 'Control Flags', value: '0x00', size: '1 byte' },
                    { name: 'Reserved', value: '0x00', size: '1 byte' }
                ],
                payload: '[TLS ENCRYPTED DATA]'
            },
            transport: {
                headers: [
                    { name: 'Source Port', value: '49152', size: '2 bytes' },
                    { name: 'Destination Port', value: '587', size: '2 bytes' },
                    { name: 'Sequence Number', value: '3847292157', size: '4 bytes' },
                    { name: 'Acknowledgment', value: '0', size: '4 bytes' },
                    { name: 'Flags', value: 'PSH, ACK', size: '2 bytes' },
                    { name: 'Window Size', value: '65535', size: '2 bytes' },
                    { name: 'Checksum', value: '0x4A2F', size: '2 bytes' },
                    { name: 'Urgent Pointer', value: '0', size: '2 bytes' }
                ],
                payload: '[SESSION + APPLICATION DATA]'
            },
            network: {
                headers: [
                    { name: 'Version', value: '4', size: '4 bits' },
                    { name: 'Header Length', value: '5 (20 bytes)', size: '4 bits' },
                    { name: 'Type of Service', value: '0x00', size: '1 byte' },
                    { name: 'Total Length', value: '256 bytes', size: '2 bytes' },
                    { name: 'Identification', value: '0x1234', size: '2 bytes' },
                    { name: 'Flags', value: 'DF (Don\'t Fragment)', size: '3 bits' },
                    { name: 'Fragment Offset', value: '0', size: '13 bits' },
                    { name: 'TTL', value: '64', size: '1 byte' },
                    { name: 'Protocol', value: '6 (TCP)', size: '1 byte' },
                    { name: 'Header Checksum', value: '0x4C7A', size: '2 bytes' },
                    { name: 'Source IP', value: this.config.sourceIP, size: '4 bytes' },
                    { name: 'Destination IP', value: this.config.destIP, size: '4 bytes' }
                ],
                payload: '[TCP SEGMENT]'
            },
            datalink: {
                headers: [
                    { name: 'Preamble', value: '10101010... (7 bytes)', size: '7 bytes' },
                    { name: 'Start Frame Delimiter', value: '10101011', size: '1 byte' },
                    { name: 'Destination MAC', value: '00:1B:44:11:3A:B7', size: '6 bytes' },
                    { name: 'Source MAC', value: '00:24:D7:23:9C:85', size: '6 bytes' },
                    { name: 'EtherType', value: '0x0800 (IPv4)', size: '2 bytes' }
                ],
                payload: '[IP PACKET]',
                trailer: [
                    { name: 'Frame Check Sequence', value: '0xABCD1234 (CRC-32)', size: '4 bytes' }
                ]
            },
            physical: {
                headers: [
                    { name: 'Signal Encoding', value: 'Manchester/NRZ', size: 'N/A' },
                    { name: 'Bit Rate', value: '1 Gbps', size: 'N/A' },
                    { name: 'Signal Levels', value: '±2.5V', size: 'N/A' },
                    { name: 'Inter-Frame Gap', value: '96 bit-times', size: '12 bytes' }
                ],
                payload: '[ETHERNET FRAME AS ELECTRICAL SIGNALS]'
            }
        };

        return structures[layer] || { headers: [], payload: data };
    }

    updateAllDisplays() {
        try {
            // Update all message displays throughout slides
            const messageElements = [
                'config-message', 'app-user-data', 'app-message-body', 'final-message-tx',
                'transmission-message', 'final-summary-message', 'celebration-message'
            ];
            messageElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = this.config.message;
            });

            // Update IP addresses
            const sourceElements = ['config-source', 'ip-src-display'];
            sourceElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = this.config.sourceIP;
            });

            const destElements = ['config-dest', 'ip-dest-display'];
            destElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = this.config.destIP;
            });

            // Update protocol stack
            let stack = this.config.protocol.toUpperCase();
            if (this.config.encryption !== 'none') {
                const encNames = {
                    'tls12': 'TLS 1.2',
                    'tls13': 'TLS 1.3',
                    'ssl3': 'SSL 3.0'
                };
                stack += '/' + encNames[this.config.encryption];
            }
            stack += '/' + this.config.transport.toUpperCase() + '/IP/';
            const mediumNames = {
                'ethernet': 'Ethernet',
                'wifi': 'Wi-Fi',
                'fiber': 'Fiber',
                'cellular': '5G'
            };
            stack += mediumNames[this.config.medium];
            
            const stackElement = document.getElementById('config-stack');
            if (stackElement) stackElement.textContent = stack;

            // Update security info
            const securityText = this.config.encryption === 'none' ? 'No Encryption' : 
                               this.config.encryption === 'tls12' ? 'TLS 1.2 Encrypted' :
                               this.config.encryption === 'tls13' ? 'TLS 1.3 Encrypted' : 'SSL 3.0 Encrypted';
            const securityElement = document.getElementById('config-security');
            if (securityElement) securityElement.textContent = securityText;

            // Update ports based on protocol
            const portMap = {
                smtp: { src: '49152', dest: '587' },
                http: { src: '49153', dest: '80' },
                ftp: { src: '49154', dest: '21' },
                dns: { src: '49155', dest: '53' }
            };
            
            const ports = portMap[this.config.protocol];
            ['tcp-src-display', 'tcp-src-port'].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = ports.src;
            });
            ['tcp-dest-display', 'tcp-dest-port'].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = ports.dest;
            });

            // Update transport protocol display safely
            const transportHeader = document.getElementById('transport-header');
            if (transportHeader) {
                transportHeader.textContent = this.config.transport.toUpperCase() + ' Header (' + (this.config.transport === 'tcp' ? '20' : '8') + ' bytes)';
            }
            
            const transportPdu = document.getElementById('transport-pdu');
            if (transportPdu) {
                transportPdu.textContent = this.config.transport.toUpperCase() + ' ' + (this.config.transport === 'tcp' ? 'Segment' : 'Datagram');
            }

            // Update IP protocol field safely
            const ipProtocol = document.getElementById('ip-protocol');
            if (ipProtocol) {
                ipProtocol.textContent = this.config.transport === 'tcp' ? '6 (TCP)' : '17 (UDP)';
            }

            // Update physical medium details safely
            const mediumDetails = {
                ethernet: {
                    name: '1000BASE-T Ethernet',
                    encoding: 'Manchester Encoding',
                    signal: '±2.5V electrical pulses',
                    speed: '1 Gbps'
                },
                wifi: {
                    name: '802.11ax Wi-Fi 6',
                    encoding: 'OFDM Modulation',
                    signal: 'Radio waves (2.4/5 GHz)',
                    speed: '9.6 Gbps'
                },
                fiber: {
                    name: '10GBASE-SR Fiber',
                    encoding: '64B/66B Encoding',
                    signal: 'Light pulses (850nm)',
                    speed: '10 Gbps'
                },
                cellular: {
                    name: '5G NR',
                    encoding: 'QAM Modulation',
                    signal: 'Radio waves (28 GHz)',
                    speed: '20 Gbps'
                }
            };

            const medium = mediumDetails[this.config.medium];
            
            // Update medium info with safety checks
            const mediumUpdates = [
                { ids: ['physical-medium', 'phys-medium'], value: medium.name },
                { ids: ['physical-encoding', 'phys-encoding'], value: medium.encoding },
                { ids: ['physical-signal', 'phys-signal'], value: medium.signal }
            ];

            mediumUpdates.forEach(update => {
                update.ids.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) element.textContent = update.value;
                });
            });

            const physicalSpeed = document.getElementById('physical-speed');
            if (physicalSpeed) physicalSpeed.textContent = medium.speed;

            // Update signal type descriptions safely
            const signalTypes = {
                ethernet: 'electrical signals',
                wifi: 'radio waves',
                fiber: 'light pulses',
                cellular: 'radio frequencies'
            };
            
            const signalType = document.getElementById('signal-type');
            if (signalType) signalType.textContent = signalTypes[this.config.medium];
            
            const mediumType = document.getElementById('medium-type');
            if (mediumType) mediumType.textContent = medium.name.toLowerCase();

            // Calculate and update sizes
            this.updateSizeCalculations();
            
            // Update enhanced data representations
            this.updateDataRepresentations();
            
        } catch (error) {
            console.error('Error updating displays:', error);
        }
    }

    updateDataRepresentations() {
        try {
            // Update binary and hex representations for each layer
            const binaryData = this.generateBinaryData(this.config.message);
            const hexData = this.generateHexData(this.config.message);
            
            // Update binary displays
            const binaryElements = document.querySelectorAll('.binary-data');
            binaryElements.forEach(el => {
                el.textContent = `Binary: ${binaryData}`;
            });
            
            // Update hex displays
            const hexElements = document.querySelectorAll('.hex-data');
            hexElements.forEach(el => {
                el.textContent = `Hex: ${hexData}`;
            });
            
            // Generate and update packet structures for each layer
            const layers = ['application', 'presentation', 'session', 'transport', 'network', 'datalink', 'physical'];
            layers.forEach(layer => {
                this.updateLayerPacketStructure(layer);
            });
            
        } catch (error) {
            console.error('Error updating data representations:', error);
        }
    }

    updateLayerPacketStructure(layer) {
        const structure = this.generatePacketStructure(layer, this.config.message);
        const containerId = `${layer}-packet-structure`;
        const container = document.getElementById(containerId);
        
        if (container) {
            let html = '<div class="data-structure">';
            
            // Add headers
            structure.headers.forEach(header => {
                html += `
                    <div class="structure-layer current">
                        <span class="layer-label">${header.name}:</span>
                        <span class="layer-data">${header.value}</span>
                        <span class="layer-size">${header.size}</span>
                    </div>
                `;
            });
            
            // Add payload
            html += `
                <div class="structure-layer">
                    <span class="layer-label">Payload:</span>
                    <span class="layer-data">${structure.payload}</span>
                    <span class="layer-size">${structure.payload.length} bytes</span>
                </div>
            `;
            
            // Add trailer if exists
            if (structure.trailer) {
                structure.trailer.forEach(trailer => {
                    html += `
                        <div class="structure-layer current">
                            <span class="layer-label">${trailer.name}:</span>
                            <span class="layer-data">${trailer.value}</span>
                            <span class="layer-size">${trailer.size}</span>
                        </div>
                    `;
                });
            }
            
            html += '</div>';
            container.innerHTML = html;
        }
    }

    updateSizeCalculations() {
        try {
            const messageSize = new TextEncoder().encode(this.config.message).length;
        
            // Application layer
            const protocolOverhead = {
                smtp: 178, http: 95, ftp: 87, dns: 45
            };
            const appHeaders = protocolOverhead[this.config.protocol];
            const appTotal = messageSize + appHeaders;
            
            // Presentation layer - compression then encryption
            const compressionRatio = 0.194; // 19.4% reduction for typical text
            const compressedSize = Math.floor(appTotal * (1 - compressionRatio));
            const compressionSavings = appTotal - compressedSize;
            
            const encryptionOverhead = {
                tls12: 28, tls13: 32, ssl3: 24, none: 0
            };
            const encryptionHeaders = encryptionOverhead[this.config.encryption];
            const presTotal = compressedSize + encryptionHeaders;
            const netPresentationChange = presTotal - appTotal; // Can be negative (savings)
            
            // Session layer (8 bytes)
            const sessTotal = presTotal + 8;
            
            // Transport layer
            const transportHeaders = this.config.transport === 'tcp' ? 20 : 8;
            const transportTotal = sessTotal + transportHeaders;
            
            // Network layer (20 bytes for IPv4)
            const networkTotal = transportTotal + 20;
            
            // Data Link layer (18 bytes for Ethernet)
            const datalinkTotal = networkTotal + 18;
            
            // Physical layer (20 bytes overhead)
            const physicalTotal = datalinkTotal + 20;

            // Safely update all size displays with null checks
            const sizeUpdates = [
                { id: 'app-overhead', value: appHeaders + ' bytes' },
                { id: 'app-total', value: appTotal + ' bytes' },
                { id: 'pres-overhead', value: netPresentationChange + ' bytes' },
                { id: 'pres-total', value: presTotal + ' bytes' },
                { id: 'sess-total', value: sessTotal + ' bytes' },
                { id: 'transport-overhead', value: transportHeaders + ' bytes' },
                { id: 'transport-total', value: transportTotal + ' bytes' },
                { id: 'network-total', value: networkTotal + ' bytes' },
                { id: 'datalink-total', value: datalinkTotal + ' bytes' },
                { id: 'physical-total', value: physicalTotal + ' bytes' },
                // Additional compression-specific updates
                { id: 'pres-total-display', value: presTotal + ' bytes' },
                { id: 'app-total-display', value: appTotal + ' bytes' }
            ];

            sizeUpdates.forEach(update => {
                const element = document.getElementById(update.id);
                if (element) element.textContent = update.value;
            });

            // Update summary displays
            const sizeElements = ['original-size', 'transmission-original', 'final-original-size'];
            sizeElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = messageSize;
            });

            const totalElements = ['total-size', 'transmission-total', 'final-total-size'];
            totalElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = physicalTotal + ' bytes';
            });

            const overhead = physicalTotal - messageSize;
            const element = document.getElementById('final-overhead');
            if (element) element.textContent = overhead + ' bytes';

            const ratio = Math.round(physicalTotal / messageSize);
            const ratioElements = ['overhead-preview', 'transmission-ratio'];
            ratioElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = ratio + ':1';
            });

            // Update compression-specific elements
            const compressionElements = document.querySelectorAll('.compression-savings');
            compressionElements.forEach(el => {
                el.textContent = `📦 Compression: -${compressionSavings} bytes`;
            });

            const encryptionElements = document.querySelectorAll('.encryption-overhead');
            encryptionElements.forEach(el => {
                el.textContent = `🔐 Encryption: +${encryptionHeaders} bytes`;
            });

            const netElements = document.querySelectorAll('.net-result');
            netElements.forEach(el => {
                const netChange = netPresentationChange >= 0 ? `+${netPresentationChange}` : `${netPresentationChange}`;
                el.textContent = `✅ Net: ${netChange} bytes saved`;
            });
        } catch (error) {
            console.error('Error calculating sizes:', error);
        }
    }

    updateSlideDisplay() {
        const slides = document.querySelectorAll('.slide');
        slides.forEach(slide => slide.classList.remove('active'));
        
        if (slides[this.currentSlide - 1]) {
            slides[this.currentSlide - 1].classList.add('active');
        }
        
        const currentSlideElement = document.getElementById('current-slide');
        if (currentSlideElement) currentSlideElement.textContent = this.currentSlide;
        
        const totalSlidesElement = document.getElementById('total-slides');
        if (totalSlidesElement) totalSlidesElement.textContent = this.totalSlides;
        
        const progress = (this.currentSlide / this.totalSlides) * 100;
        const progressFill = document.getElementById('progress-fill');
        if (progressFill) progressFill.style.width = progress + '%';
        
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        if (prevBtn) prevBtn.disabled = this.currentSlide === 1;
        if (nextBtn) nextBtn.disabled = this.currentSlide === this.totalSlides;
        
        if (nextBtn) {
            nextBtn.textContent = this.currentSlide === this.totalSlides ? 'Complete' : 'Next';
        }
    }

    changeSlide(direction) {
        const newSlide = this.currentSlide + direction;
        
        if (newSlide >= 1 && newSlide <= this.totalSlides) {
            this.currentSlide = newSlide;
            this.updateSlideDisplay();
        }
    }

    restartSlideshow() {
        this.currentSlide = 1;
        this.updateSlideDisplay();
    }

    setupEventListeners() {
        // Update button
        const updateBtn = document.getElementById('updateBtn');
        if (updateBtn) {
            updateBtn.addEventListener('click', () => this.updateConfiguration());
        }

        // Navigation buttons
        this.setupNavigationButtons();
    }

    setupNavigationButtons() {
        try {
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const restartBtn = document.querySelector('.nav-btn.restart');
            
            if (prevBtn) {
                prevBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.changeSlide(-1);
                });
            }
            
            if (nextBtn) {
                nextBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.changeSlide(1);
                });
            }
            
            if (restartBtn) {
                restartBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.restartSlideshow();
                });
            }
        } catch (error) {
            console.error('Error setting up navigation buttons:', error);
        }
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (event) => {
            try {
                if (this.keyboardDebounce) return;
                this.keyboardDebounce = true;
                
                if (event.key === 'ArrowLeft') {
                    this.changeSlide(-1);
                } else if (event.key === 'ArrowRight') {
                    this.changeSlide(1);
                } else if (event.key === 'Home') {
                    this.restartSlideshow();
                }
                
                setTimeout(() => {
                    this.keyboardDebounce = false;
                }, 100);
            } catch (error) {
                console.error('Keyboard navigation error:', error);
                this.keyboardDebounce = false;
            }
        });
    }

    // Initialize interactive tooltips
    initializeTooltips() {
        const tooltips = document.querySelectorAll('.tooltip');
        
        tooltips.forEach(tooltip => {
            tooltip.addEventListener('mouseenter', (e) => {
                this.showTooltip(e.target);
            });
            
            tooltip.addEventListener('mouseleave', (e) => {
                this.hideTooltip();
            });
            
            tooltip.addEventListener('mousemove', (e) => {
                this.positionTooltip(e);
            });
        });
    }

    showTooltip(element) {
        // Remove any existing tooltip
        this.hideTooltip();
        
        const tooltipText = element.getAttribute('data-tooltip');
        if (!tooltipText) return;
        
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-popup';
        tooltip.textContent = tooltipText;
        tooltip.id = 'active-tooltip';
        
        document.body.appendChild(tooltip);
        
        // Add slight delay for smooth appearance
        setTimeout(() => {
            tooltip.classList.add('visible');
        }, 50);
    }

    hideTooltip() {
        const existingTooltip = document.getElementById('active-tooltip');
        if (existingTooltip) {
            existingTooltip.remove();
        }
    }

    positionTooltip(e) {
        const tooltip = document.getElementById('active-tooltip');
        if (!tooltip) return;
        
        const rect = tooltip.getBoundingClientRect();
        const padding = 10;
        
        let x = e.clientX + padding;
        let y = e.clientY - rect.height - padding;
        
        // Adjust if tooltip would go off screen
        if (x + rect.width > window.innerWidth) {
            x = e.clientX - rect.width - padding;
        }
        
        if (y < 0) {
            y = e.clientY + padding;
        }
        
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
    }

    safeInitialize() {
        try {
            // Set current date safely
            const currentDateElement = document.getElementById('current-date');
            if (currentDateElement) {
                currentDateElement.textContent = new Date().toUTCString();
            }

            // Initialize displays
            this.updateAllDisplays();
            this.updateSlideDisplay();
            
            console.log('OSI Model slideshow initialized successfully');
        } catch (error) {
            console.error('Initialization error:', error);
            // Continue with basic functionality even if some elements are missing
            this.updateSlideDisplay();
        }
    }
}

// Global functions for backward compatibility
let osiAnimation;

function updateConfiguration() {
    if (osiAnimation) {
        osiAnimation.updateConfiguration();
    }
}

function changeSlide(direction) {
    if (osiAnimation) {
        osiAnimation.changeSlide(direction);
    }
}

function restartSlideshow() {
    if (osiAnimation) {
        osiAnimation.restartSlideshow();
    }
}

// Initialize when DOM is ready
function initializeWhenReady() {
    console.log('Starting OSI Model initialization...');
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, creating OSI animation...');
            try {
                osiAnimation = new OSIModelAnimation();
                console.log('OSI animation created successfully');
            } catch (error) {
                console.error('Error creating OSI animation:', error);
            }
        });
    } else {
        // DOM is already ready
        console.log('DOM already ready, creating OSI animation...');
        setTimeout(() => {
            try {
                osiAnimation = new OSIModelAnimation();
                console.log('OSI animation created successfully');
            } catch (error) {
                console.error('Error creating OSI animation:', error);
            }
        }, 100);
    }
}

// Start initialization
console.log('OSI Model script loaded');
initializeWhenReady();

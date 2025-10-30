"""
Utilities module for OpenScope RL environment.

This module provides shared utilities for browser management,
JavaScript injection, and common helper functions.
"""

import time
import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from playwright.sync_api import Browser, Page, Playwright, sync_playwright
from playwright.async_api import async_playwright, Browser as AsyncBrowser, Page as AsyncPage, Playwright as AsyncPlaywright

from .config import BrowserConfig
from .exceptions import BrowserError, GameInterfaceError
from .constants import JS_GET_OPTIMAL_GAME_STATE_SCRIPT


logger = logging.getLogger(__name__)


class PageWrapper:
    """
    Wrapper class that provides a unified sync interface for both sync and async Playwright pages.

    This allows the rest of the codebase to remain synchronous while supporting
    Jupyter notebook environments that require async Playwright.
    """

    def __init__(self, page: Union[Page, AsyncPage], is_async: bool):
        """
        Initialize page wrapper.

        Args:
            page: Playwright page (sync or async)
            is_async: Whether the page is async
        """
        self._page = page
        self._is_async = is_async

    def _run_async(self, coro):
        """Helper to run async coroutines."""
        if self._is_async:
            # In Jupyter, we're already in an event loop
            # We need to use await, but since we're in a sync method,
            # we create a task and wait for it synchronously
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)
        return coro

    def evaluate(self, script, arg=None):
        """Execute JavaScript and return result."""
        if self._is_async:
            if arg is not None:
                return self._run_async(self._page.evaluate(script, arg))
            return self._run_async(self._page.evaluate(script))
        else:
            if arg is not None:
                return self._page.evaluate(script, arg)
            return self._page.evaluate(script)

    def goto(self, url, **kwargs):
        """Navigate to URL."""
        if self._is_async:
            return self._run_async(self._page.goto(url, **kwargs))
        return self._page.goto(url, **kwargs)

    def wait_for_load_state(self, state="load", **kwargs):
        """Wait for load state."""
        if self._is_async:
            return self._run_async(self._page.wait_for_load_state(state, **kwargs))
        return self._page.wait_for_load_state(state, **kwargs)

    def add_init_script(self, script):
        """Add initialization script."""
        if self._is_async:
            return self._run_async(self._page.add_init_script(script))
        return self._page.add_init_script(script)

    def close(self):
        """Close the page."""
        if self._is_async:
            return self._run_async(self._page.close())
        return self._page.close()

    def on(self, event, handler):
        """Register event handler."""
        return self._page.on(event, handler)

    @property
    def url(self):
        """Get current URL."""
        return self._page.url

    def title(self):
        """Get page title."""
        if self._is_async:
            return self._run_async(self._page.title())
        return self._page.title()

    def fill(self, selector: str, value: str, **kwargs):
        """Fill an input field."""
        if self._is_async:
            return self._run_async(self._page.fill(selector, value, **kwargs))
        return self._page.fill(selector, value, **kwargs)
    
    def press(self, selector: str, key: str, **kwargs):
        """Press a key on an element."""
        if self._is_async:
            return self._run_async(self._page.press(selector, key, **kwargs))
        return self._page.press(selector, key, **kwargs)
    
    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped page."""
        return getattr(self._page, name)


class BrowserManager:
    """
    Manages Playwright browser lifecycle and page operations.
    
    This class handles browser initialization, page creation, and cleanup
    using the context manager protocol for proper resource management.
    """
    
    def __init__(self, config: BrowserConfig):
        """
        Initialize browser manager.

        Args:
            config: Browser configuration settings
        """
        self.config = config
        self.playwright: Optional[Union[Playwright, AsyncPlaywright]] = None
        self.browser: Optional[Union[Browser, AsyncBrowser]] = None
        self.page: Optional[PageWrapper] = None  # Now always wrapped
        self._is_initialized = False
        self._is_async = False  # Track if using async API
    
    def __enter__(self) -> 'BrowserManager':
        """Enter context manager."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup resources."""
        self.cleanup()
    
    def initialize(self) -> None:
        """
        Initialize browser and create page.

        Raises:
            BrowserError: If browser initialization fails
        """
        try:
            logger.info("Initializing browser...")

            # Check if we're in an asyncio event loop (e.g., Jupyter notebook)
            try:
                loop = asyncio.get_running_loop()
                in_event_loop = True
                logger.info("Detected running asyncio event loop (Jupyter/notebook environment)")
            except RuntimeError:
                in_event_loop = False

            if in_event_loop:
                # We're in an event loop (Jupyter), need to run sync playwright in a thread
                logger.info("Using thread-based workaround for Jupyter compatibility")
                self._initialize_in_thread()
            else:
                # Normal initialization
                self._initialize_direct()

            logger.info("Browser initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self.cleanup()
            raise BrowserError(f"Browser initialization failed: {e}") from e

    def _initialize_direct(self) -> None:
        """Initialize browser directly (when not in event loop)."""
        self.playwright = sync_playwright().start()

        self.browser = self.playwright.chromium.launch(
            headless=self.config.headless,
            args=self.config.browser_args
        )

        raw_page = self.browser.new_page()
        self.page = PageWrapper(raw_page, is_async=False)
        self._setup_error_handling()
        self._is_initialized = True

    def _initialize_in_thread(self) -> None:
        """Initialize browser using async API (for Jupyter/asyncio environments)."""
        # Use nest_asyncio to allow nested event loops in Jupyter
        import nest_asyncio
        nest_asyncio.apply()

        loop = asyncio.get_running_loop()

        async def async_init():
            """Async initialization function."""
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.config.headless,
                args=self.config.browser_args
            )
            raw_page = await self.browser.new_page()
            self.page = PageWrapper(raw_page, is_async=True)
            self._is_async = True

        # Run the async initialization
        loop.run_until_complete(async_init())
        self._setup_error_handling()
        self._is_initialized = True
    
    def _run_async(self, coro):
        """Helper to run async coroutines when using async API."""
        if self._is_async:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)
        return coro  # Return as-is for sync API

    def _setup_error_handling(self) -> None:
        """Setup JavaScript error handling."""
        if self.page:
            self.page.on("pageerror", lambda err: logger.error(f"JavaScript error: {err}"))
    
    def inject_time_tracking_script(self) -> None:
        """
        Inject JavaScript for tracking game time.

        This script tracks the accumulated game time using requestAnimationFrame
        to provide accurate timing for the RL environment.
        """
        if not self.page:
            raise BrowserError("Page not initialized")

        script = """
        (function() {
            let startTime = null;
            let accumulatedGameTime = 0;
            const originalRAF = window.requestAnimationFrame;

            window.requestAnimationFrame = function(callback) {
                return originalRAF.call(window, function(timestamp) {
                    if (startTime === null) startTime = timestamp;
                    accumulatedGameTime = (timestamp - startTime) / 1000;
                    return callback(timestamp);
                });
            };

            window._getRLGameTime = () => accumulatedGameTime;
        })();
        """

        try:
            self.page.add_init_script(script)  # PageWrapper handles sync/async
            logger.debug("Time tracking script injected")
        except Exception as e:
            logger.error(f"Failed to inject time tracking script: {e}")
            raise BrowserError(f"Script injection failed: {e}") from e
    
    def inject_event_hook_script(self) -> None:
        """
        Inject JavaScript to hook OpenScope scoring events.

        This patches GameController.events_recordNew to buffer all scoring
        events in window._rlEventBuffer and exposes window._rlDrainEvents()
        to retrieve and clear the buffer.
        """
        if not self.page:
            raise BrowserError("Page not initialized")

        script = """
        (() => {
          const wrappedControllers = new WeakSet();

          const ensureBuffers = () => {
            if (!window._rlEventCounts) window._rlEventCounts = {};
            if (!window._rlEventBuffer) window._rlEventBuffer = [];
          };

          const wrapController = (controller) => {
            if (!controller || wrappedControllers.has(controller)) return false;
            if (typeof controller.events_recordNew !== 'function') return false;

            try {
              const original = controller.events_recordNew;
              controller._rlEventsRecordNewOriginal = original;
              controller.events_recordNew = function patchedEventsRecordNew(evt) {
                try {
                  ensureBuffers();
                  window._rlEventCounts[evt] = (window._rlEventCounts[evt] || 0) + 1;
                  window._rlEventBuffer.push({ event: evt, ts: Date.now() });
                } catch (bufferErr) {
                  // ignore buffering failures, still forward to original handler
                }
                return original.call(this, evt);
              };
              wrappedControllers.add(controller);
              return true;
            } catch (wrapErr) {
              return false;
            }
          };

          const tryWrap = (candidate) => {
            if (!candidate) return false;
            if (Array.isArray(candidate)) {
              return candidate.reduce((acc, item) => wrapController(item) || acc, false);
            }
            return wrapController(candidate);
          };

          const probeSources = () => {
            let wrappedAny = false;
            try { wrappedAny = tryWrap(window.gameController) || wrappedAny; } catch (e) {}
            try { wrappedAny = tryWrap(window.app && window.app.gameController) || wrappedAny; } catch (e) {}
            try {
              if (window.GameController && typeof window.GameController === 'object') {
                wrappedAny = tryWrap(window.GameController.instance || window.GameController) || wrappedAny;
              }
            } catch (e) {}
            return wrappedAny;
          };

          const hookProperty = (host, prop) => {
            if (!host) return;
            const descriptor = Object.getOwnPropertyDescriptor(host, prop);
            if (descriptor && descriptor.configurable === false) {
              // non-configurable property, fallback to polling
              return;
            }

            let current = host[prop];
            Object.defineProperty(host, prop, {
              configurable: true,
              get() { return current; },
              set(value) {
                current = value;
                tryWrap(value);
              }
            });

            tryWrap(current);
          };

          ensureBuffers();
          probeSources();
          hookProperty(window, 'gameController');
          if (window.app) {
            hookProperty(window.app, 'gameController');
          }

          if (!window._rlDrainEvents) {
            window._rlDrainEvents = () => {
              ensureBuffers();
              const out = {
                events: window._rlEventBuffer.slice(),
                counts: { ...window._rlEventCounts }
              };
              window._rlEventBuffer.length = 0;
              return out;
            };
          }

          const pollingInterval = setInterval(() => {
            const wrapped = probeSources();
            if (wrappedControllers.size > 0 && wrapped) {
              clearInterval(pollingInterval);
              clearTimeout(stopPollingTimer);
            }
          }, 250);

          const stopPollingTimer = setTimeout(() => clearInterval(pollingInterval), 15000);
        })();
        """

        try:
            self.page.add_init_script(script)
            logger.debug("Event hook script injected")
        except Exception as e:
            logger.error(f"Failed to inject event hook script: {e}")
            raise BrowserError(f"Script injection failed: {e}") from e
    
    def navigate_to_game(self, game_url: str) -> None:
        """
        Navigate to the OpenScope game URL.

        Args:
            game_url: URL of the OpenScope game

        Raises:
            BrowserError: If navigation fails
        """
        if not self.page:
            raise BrowserError("Page not initialized")

        try:
            logger.info(f"Navigating to {game_url}")

            # PageWrapper handles sync/async
            response = self.page.goto(game_url, wait_until="networkidle", timeout=30000)
            if response and response.status >= 400:
                logger.error(f"Navigation failed with status {response.status}")
                raise BrowserError(f"Navigation failed with HTTP {response.status}")

            self.page.wait_for_load_state("networkidle")

            # Additional wait to ensure JavaScript has loaded
            time.sleep(3)

            # Verify navigation was successful
            current_url = self.page.url
            logger.info(f"Navigation completed. Current URL: {current_url}")

            # Check if we're still on about:blank
            if current_url == "about:blank":
                logger.warning("Navigation resulted in about:blank page, retrying...")
                self.page.goto(game_url, wait_until="domcontentloaded", timeout=30000)
                self.page.wait_for_load_state("networkidle")
                time.sleep(5)  # Longer wait

                current_url = self.page.url
                logger.info(f"Retry navigation completed. Current URL: {current_url}")

                if current_url == "about:blank":
                    raise BrowserError("Navigation consistently results in about:blank page")

            # Verify we can access the page content
            try:
                title = self.page.title()
                logger.info(f"Page title: {title}")
            except Exception as e:
                logger.warning(f"Could not get page title: {e}")

        except Exception as e:
            logger.error(f"Failed to navigate to game: {e}")
            raise BrowserError(f"Navigation failed: {e}") from e
    
    def wait_for_game_update(self, num_frames: int = 1) -> None:
        """
        Wait for OpenScope to process game updates.
        
        Args:
            num_frames: Number of frames to wait
        """
        if not self.page:
            raise BrowserError("Page not initialized")
        
        # Wait for frame processing
        time.sleep(num_frames * self.config.game_update_delay)
    
    def cleanup(self) -> None:
        """Clean up browser resources."""
        try:
            if self._is_async:
                # Async cleanup with nest_asyncio support
                async def async_cleanup():
                    if self.page:
                        await self.page._page.close()  # Access wrapped page
                    if self.browser:
                        await self.browser.close()
                    if self.playwright:
                        await self.playwright.stop()

                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop = asyncio.get_running_loop()
                    loop.run_until_complete(async_cleanup())
                except RuntimeError:
                    # No event loop running anymore, can't cleanup async resources
                    logger.warning("No event loop for async cleanup, resources may leak")
            else:
                # Sync cleanup
                if self.page:
                    self.page.close()  # PageWrapper handles it
                if self.browser:
                    self.browser.close()
                if self.playwright:
                    self.playwright.stop()

            self.page = None
            self.browser = None
            self.playwright = None
            self._is_initialized = False
            self._is_async = False
            logger.info("Browser resources cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if browser is initialized."""
        return self._is_initialized and self.page is not None


def execute_command(page: Page, command: str) -> None:
    """
    Execute a command in the OpenScope game using Playwright's native methods.
    
    Uses Playwright's built-in fill() and press() methods which are more reliable
    than custom JavaScript event dispatching. These methods properly trigger all
    necessary DOM events and work correctly with frameworks like jQuery.
    
    Args:
        page: Playwright page object (supports both sync Page and PageWrapper)
        command: Command string to execute
        
    Raises:
        GameInterfaceError: If command execution fails
    """
    if not page:
        raise GameInterfaceError("Page not available")
    
    try:
        # Use Playwright's native methods for more reliable command execution
        # fill() properly triggers input events and press() handles key events correctly
        # Works with both sync Page and async PageWrapper
        page.fill('#command', command)
        page.press('#command', 'Enter')
        logger.debug(f"Executed command: {command}")
        
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}")
        raise GameInterfaceError(f"Command execution failed: {e}") from e


def extract_game_state(page: Page) -> Dict[str, Any]:
    """
    Extract complete game state from OpenScope.
    
    This is the primary state extraction function for the environment. It extracts:
    - The 14 core features per aircraft needed by StateProcessor (position, altitude, heading, etc.)
    - Additional ATC-critical data: wind components, flight phase, waypoints, flight plan,
      approach clearance status, and landing status
    - Global state: score, time, number of aircraft
    - Conflicts: distance, altitude separation, violation status
    - Scoring events: captured via injected event hook (events, event_counts)
    
    Args:
        page: Playwright page object
        
    Returns:
        Dict containing:
            - aircraft: List of aircraft with full state
            - conflicts: List of active conflicts/violations
            - score: Current game score
            - time: Game time in seconds
            - numAircraft: Number of active aircraft
            - events: List of recent scoring events (if event hook is active)
            - event_counts: Cumulative counts of each event type (if event hook is active)
        
    Raises:
        GameInterfaceError: If state extraction fails
    """
    if not page:
        raise GameInterfaceError("Page not available")
    
    try:
        result = page.evaluate(JS_GET_OPTIMAL_GAME_STATE_SCRIPT)
        # Drain any buffered scoring events if hook is present
        try:
            drain = page.evaluate("() => window._rlDrainEvents ? window._rlDrainEvents() : null")
            if isinstance(result, dict) and isinstance(drain, dict):
                events = drain.get("events") or []
                counts = drain.get("counts") or {}
                if events or counts:
                    result["events"] = events
                    result["event_counts"] = counts
        except Exception:
            pass
        
        if result is None:
            logger.warning("Game state extraction returned null")
            return {}
        
        logger.debug(f"Extracted state: {len(result.get('aircraft', []))} aircraft, "
                    f"{len(result.get('conflicts', []))} conflicts")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract game state: {e}")
        raise GameInterfaceError(f"State extraction failed: {e}") from e


# Backwards compatibility alias
extract_optimal_game_state = extract_game_state


 


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available PyTorch device, supporting CUDA, Metal (MPS), and CPU.
    
    Priority order:
    1. Explicitly provided device
    2. CUDA (if available)
    3. Metal/MPS (if available on Apple Silicon)
    4. CPU (fallback)
    
    Args:
        device: Optional device string (e.g., "cuda", "mps", "cpu"). 
                If None, auto-detects the best available device.
        
    Returns:
        Device string suitable for torch.device()
        
    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        >>> model = model.to(torch.device(device))
    """
    import torch
    
    # If device is explicitly provided, use it
    if device is not None:
        return device
    
    # Check CUDA availability (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"
    
    # Check Metal availability (Apple Silicon GPUs)
    # PyTorch uses "mps" as the device name for Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # Fallback to CPU
    return "cpu"